"""torchmodel: recover a libtorch (TorchScript) model from a Linux process.

Walks  c10::ivalue::Object -> slots_ -> ClassType -> attributes_  to rebuild the
module tree (layer names, classes, order) and follows TensorImpl -> StorageImpl
-> data to extract the weights as .npy files.

Struct offsets below were verified against PyTorch 2.14.1 (x86_64, Linux) with
gdb and volshell. They are NOT portable across torch versions.
"""
import json
import logging
import re
import struct
from typing import Dict, Iterator, List, Optional

from volatility3.framework import exceptions, interfaces, renderers
from volatility3.framework.configuration import requirements
from volatility3.framework.renderers import format_hints
from volatility3.plugins.linux import pslist

vollog = logging.getLogger(__name__)

# ---- offsets (PyTorch 2.14.1) -------------------------------------------
OBJ_TYPE = 0x40        # ivalue::Object::type_  (raw ClassType* is first word)
OBJ_SLOTS = 0x50       # ivalue::Object::slots_ (vector<IValue>: begin,end,cap)
IVALUE_SIZE = 16       # payload (8) + tag (low 4 bytes of 2nd word)
TAG_TENSOR = 0x1
TAG_OBJECT = 0x14

CT_NAME = 0x38         # ClassType::name_ -> std::string qualified name
CT_ATTRS = 0xE0        # ClassType::attributes_ (vector<ClassAttribute>)
ATTR_SIZE = 0x38       # sizeof(ClassAttribute): kind(8) type(16) name(32)
ATTR_KIND = 0x00
ATTR_NAME = 0x18

TI_STORAGE = 0x10      # TensorImpl::storage_ (-> StorageImpl*)
TI_NDIM = 0x38         # SizesAndStrides::size_
TI_SIZES = 0x40        # inline sizes[5]
TI_STRIDES = 0x68      # inline strides[5]
SI_DATA = 0x10         # StorageImpl::data_ptr
MAX_INLINE_DIMS = 5

ATTR_KINDS = {0: "buffer", 1: "parameter", 2: "attribute"}
CHUNK = 1 << 20


class Mem:
    """Small helper around a Volatility layer for typed reads."""

    def __init__(self, layer):
        self.layer = layer

    def read(self, addr: int, size: int) -> bytes:
        return self.layer.read(addr, size, pad=False)

    def u64(self, addr: int) -> int:
        return struct.unpack("<Q", self.read(addr, 8))[0]

    def s64(self, addr: int) -> int:
        return struct.unpack("<q", self.read(addr, 8))[0]

    def u32(self, addr: int) -> int:
        return struct.unpack("<I", self.read(addr, 4))[0]

    def stdstr(self, addr: int) -> str:
        ptr, size = self.u64(addr), self.u64(addr + 8)
        if size > 256:
            raise ValueError("implausible std::string size")
        return self.read(ptr, size).decode("utf-8", "replace")


def npy_bytes(raw: bytes, shape) -> bytes:
    """Minimal .npy (v1.0) writer for little-endian float32."""
    dims = ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "")
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }" % dims
    pad = (64 - (10 + len(header) + 1) % 64) % 64
    header += " " * pad + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + raw


class TorchModel(interfaces.plugins.PluginInterface):
    """Recovers a TorchScript model (architecture + weights) from a process."""

    _required_framework_version = (2, 0, 0)
    _version = (0, 1, 0)

    @classmethod
    def get_requirements(cls):
        return [
            requirements.ModuleRequirement(
                name="kernel",
                description="Linux kernel",
                architectures=["Intel32", "Intel64"],
            ),
            requirements.IntRequirement(
                name="pid", description="PID of the inference process", optional=False
            ),
            requirements.StringRequirement(
                name="exe",
                description="Name of the executable VMA (default cifar_inference)",
                optional=True,
                default="cifar_inference",
            ),
            requirements.StringRequirement(
                name="vtoff",
                description="File offset of 'vtable for c10::ivalue::Object' "
                "in the executable (from nm), e.g. 0x3b7e8",
                optional=True,
                default="0x3b7e8",
            ),
            requirements.StringRequirement(
                name="module",
                description="Skip scanning: address of the root ivalue::Object",
                optional=True,
            ),
            requirements.BooleanRequirement(
                name="dump",
                description="Write weights (.npy) and architecture.json",
                default=False,
                optional=True,
            ),
        ]

    # ------------------------------------------------------------------
    def _find_task(self, pid: int):
        flt = pslist.PsList.create_pid_filter([pid])
        for task in pslist.PsList.list_tasks(self.context, self.config["kernel"], flt):
            return task
        return None

    @staticmethod
    def _vmas(context, task):
        out = []
        for vma in task.mm.get_vma_iter():
            try:
                name = vma.get_name(context, task) or ""
            except Exception:
                name = ""
            try:
                prot = str(vma.get_protection())
            except Exception:
                prot = "???"
            out.append((int(vma.vm_start), int(vma.vm_end), name, prot))
        return out

    # ---- discovery -----------------------------------------------------
    @staticmethod
    def _scan(layer, vmas, needle: bytes) -> Iterator[int]:
        for start, end, name, prot in vmas:
            if not (name == "[heap]" or (name == "" and prot.startswith("rw"))):
                continue
            pos = start
            while pos < end:
                size = min(CHUNK, end - pos)
                data = layer.read(pos, size, pad=True)
                idx = data.find(needle)
                while idx != -1:
                    if idx % 8 == 0:
                        yield pos + idx
                    idx = data.find(needle, idx + 1)
                pos += size

    # ---- parsing -------------------------------------------------------
    @staticmethod
    def _parse_module(mem: Mem, obj: int) -> Dict:
        ct = mem.u64(obj + OBJ_TYPE)
        cls = mem.stdstr(ct + CT_NAME)
        if not cls.startswith("__torch__"):
            raise ValueError("not a TorchScript class")
        sb, se = mem.u64(obj + OBJ_SLOTS), mem.u64(obj + OBJ_SLOTS + 8)
        if se < sb or (se - sb) % IVALUE_SIZE or (se - sb) // IVALUE_SIZE > 4096:
            raise ValueError("bad slots_")
        nslots = (se - sb) // IVALUE_SIZE
        ab, ae = mem.u64(ct + CT_ATTRS), mem.u64(ct + CT_ATTRS + 8)
        if ae < ab or (ae - ab) % ATTR_SIZE or (ae - ab) // ATTR_SIZE != nslots:
            raise ValueError("attributes_/slots_ mismatch")
        slots = []
        for i in range(nslots):
            payload = mem.u64(sb + IVALUE_SIZE * i)
            tag = mem.u32(sb + IVALUE_SIZE * i + 8)
            a = ab + ATTR_SIZE * i
            slots.append(
                {
                    "index": i,
                    "name": mem.stdstr(a + ATTR_NAME),
                    "kind": ATTR_KINDS.get(mem.u64(a + ATTR_KIND) & 0xFFFFFFFF, "?"),
                    "tag": tag,
                    "payload": payload,
                }
            )
        return {"addr": obj, "class": cls, "slots": slots}

    @staticmethod
    def _parse_tensor(mem: Mem, t: int) -> Dict:
        nd = mem.u64(t + TI_NDIM)
        if nd > MAX_INLINE_DIMS:
            raise ValueError("ndim %d not stored inline" % nd)
        sizes = [mem.s64(t + TI_SIZES + 8 * i) for i in range(nd)]
        strides = [mem.s64(t + TI_STRIDES + 8 * i) for i in range(nd)]
        storage = mem.u64(t + TI_STORAGE)
        data = mem.u64(storage + SI_DATA)
        numel = 1
        for s in sizes:
            numel *= s
        exp, acc = [], 1
        for s in reversed(sizes):
            exp.append(acc)
            acc *= max(s, 1)
        return {
            "addr": t,
            "sizes": sizes,
            "strides": strides,
            "contiguous": strides == list(reversed(exp)),
            "data": data,
            "numel": numel,
        }

    # ---- tree building -------------------------------------------------
    def _build(self, mem: Mem, obj: int, path: str, seen: set) -> Dict:
        node = self._parse_module(mem, obj)
        node["path"] = path
        node["children"], node["tensors"], node["other"] = [], [], []
        seen.add(obj)
        for s in node["slots"]:
            sp = path + "." + s["name"]
            if s["tag"] == TAG_OBJECT and s["payload"] not in seen:
                try:
                    node["children"].append(self._build(mem, s["payload"], sp, seen))
                except (ValueError, exceptions.InvalidAddressException) as e:
                    vollog.debug("child %s failed: %s", sp, e)
            elif s["tag"] == TAG_TENSOR:
                try:
                    t = self._parse_tensor(mem, s["payload"])
                    t.update(name=s["name"], path=sp, kind=s["kind"])
                    node["tensors"].append(t)
                except (ValueError, exceptions.InvalidAddressException) as e:
                    vollog.debug("tensor %s failed: %s", sp, e)
            else:
                node["other"].append(s)
        return node

    def _child_objects(self, mem: Mem, obj: int) -> List[int]:
        try:
            m = self._parse_module(mem, obj)
        except (ValueError, struct.error, exceptions.InvalidAddressException):
            return []
        return [s["payload"] for s in m["slots"] if s["tag"] == TAG_OBJECT]

    # ---- output --------------------------------------------------------
    def _dump_tensor(self, mem: Mem, pid: int, t: Dict) -> str:
        nbytes = t["numel"] * 4  # float32 assumed (verify against model)
        note = ""
        try:
            raw = mem.read(t["data"], nbytes)
        except exceptions.InvalidAddressException:
            raw = mem.layer.read(t["data"], nbytes, pad=True)
            note = "PARTIAL (missing pages zero-filled)"
        fname = "pid%d.%s.npy" % (pid, re.sub(r"[^A-Za-z0-9_.]", "_", t["path"]))
        try:
            with self.open(fname) as fh:
                fh.write(npy_bytes(raw, t["sizes"]))
        except Exception as e:
            vollog.error("could not write %s: %s", fname, e)
            return "WRITE FAILED"
        return (fname + " " + note).strip()

    def _flatten(self, node, depth=0):
        yield depth, node["path"], node["class"], "", node["addr"], 0, 0, ""
        for t in node["tensors"]:
            yield depth + 1, t["path"], t["kind"] + " tensor", str(t["sizes"]), t["addr"], t["data"], t["numel"] * 4, t
        for c in node["children"]:
            yield from self._flatten(c, depth + 1)

    def _generator(self):
        pid = self.config["pid"]
        task = self._find_task(pid)
        if task is None or not task.mm:
            vollog.error("PID %d not found (or has no mm)", pid)
            return
        layer_name = task.add_process_layer()
        if layer_name is None:
            vollog.error("could not build process layer")
            return
        mem = Mem(self.context.layers[layer_name])
        vmas = self._vmas(self.context, task)

        if self.config.get("module"):
            roots = [int(self.config["module"], 0)]
        else:
            exe = self.config["exe"]
            bases = [v[0] for v in vmas if v[2].endswith(exe)]
            if not bases:
                vollog.error("no VMA named %s", exe)
                return
            vt = min(bases) + int(self.config["vtoff"], 0) + 0x10
            vollog.info("scanning for ivalue::Object vtable %#x", vt)
            cands = list(self._scan(mem.layer, vmas, struct.pack("<Q", vt)))
            vollog.info("%d candidate objects", len(cands))
            children = set()
            valid = []
            for c in cands:
                kids = self._child_objects(mem, c)
                if kids or self._try(mem, c):
                    valid.append(c)
                    children.update(kids)
            roots = [c for c in valid if c not in children]

        for r in roots:
            try:
                tree = self._build(mem, r, "module", set())
            except (ValueError, struct.error, exceptions.InvalidAddressException) as e:
                vollog.debug("root %#x failed: %s", r, e)
                continue
            arch = []
            for depth, path, typ, shape, obj, data, nb, t in self._flatten(tree):
                note = ""
                if isinstance(t, dict):
                    if not t["contiguous"]:
                        note = "non-contiguous"
                    if self.config.get("dump"):
                        note = (note + " " + self._dump_tensor(mem, pid, t)).strip()
                arch.append({"path": path, "type": typ, "shape": shape, "object": hex(obj), "data": hex(data) if data else None, "bytes": nb})
                yield (depth, (path, typ, shape, format_hints.Hex(obj), format_hints.Hex(data), nb, note))
            if self.config.get("dump"):
                try:
                    with self.open("pid%d.architecture.json" % pid) as fh:
                        fh.write(json.dumps({"root": hex(r), "layers": arch}, indent=2).encode())
                except Exception as e:
                    vollog.error("could not write architecture.json: %s", e)

    def _try(self, mem: Mem, obj: int) -> bool:
        try:
            self._parse_module(mem, obj)
            return True
        except (ValueError, struct.error, exceptions.InvalidAddressException):
            return False

    def run(self):
        return renderers.TreeGrid(
            [
                ("Path", str),
                ("Type", str),
                ("Shape", str),
                ("Object", format_hints.Hex),
                ("Data", format_hints.Hex),
                ("Bytes", int),
                ("Note", str),
            ],
            self._generator(),
        )
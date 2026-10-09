import os
import struct
from volatility3.framework import contexts, automagic
from volatility3.plugins.linux import psscan

# 1. Initialize Volatility 3 Context
ctx = contexts.Context()
base_config_path = "configuration"
dump_path = os.path.abspath("lime_dump/cifar_ram.raw")
ctx.config[base_config_path + ".single_location"] = "file://" + dump_path

# 2. Run automagic with (automagics, context, configurable, config_path)
automagics = automagic.available(ctx)
automagic.run(automagics, ctx, psscan.PsScan, base_config_path)

# 3. Deliverable 1.1: Locate cifar_inference (PID 40453)
plugin = psscan.PsScan(ctx, base_config_path)
task = next((t for t in plugin._generator() if t.tgid == 40453 and t.comm == "cifar_inference"), None)

if not task:
    print("[-] Process cifar_inference (PID 40453) not found.")
    exit(1)

print(f"[+] Task Found:     {task.comm}")
print(f"[+] Process PID:    {task.tgid}")
print(f"[+] Physical Offset:{hex(task.vol.offset)}")
print(f"[+] mm.pgd (CR3):   {hex(task.mm.pgd)}\n")

# 4. Deliverable 1.2: Enumerate Process VMAs
print(f"{'VMA Start':<18} {'VMA End':<18} {'Flags':<8} Mapping")
print("-" * 65)
for vma in task.mm.get_vma_iter():
    start = hex(vma.vm_start)
    end = hex(vma.vm_end)
    flags = str(vma.vm_flags)
    mapping = vma.vm_file.dentry.d_name.name if vma.vm_file else "[heap/anon]"
    print(f"{start:<18} {end:<18} {flags:<8} {mapping}")

# 5. Deliverable 2: Verify Object Layout in Memory
proc_layer_name = task.add_process_layer()
proc_layer = ctx.layers[proc_layer_name]

print("\n--- Object Memory Layout Verification ---")

# Step 5.1: Read Slot 0 to find TensorImpl
slot0_vaddr = 0x555558a35ba0
slot0_bytes = proc_layer.read(slot0_vaddr, 8)
tensor_impl = struct.unpack("<Q", slot0_bytes)[0]
print(f"1. Slot 0 ({hex(slot0_vaddr)})  -> TensorImpl:  {hex(tensor_impl)}")

# Step 5.2: Read TensorImpl + 0x10 to find StorageImpl
storage_bytes = proc_layer.read(tensor_impl + 0x10, 8)
storage_impl = struct.unpack("<Q", storage_bytes)[0]
print(f"2. TensorImpl + 0x10           -> StorageImpl: {hex(storage_impl)}")

# Step 5.3: Read StorageImpl + 0x10 to find the raw float data buffer
data_ptr_bytes = proc_layer.read(storage_impl + 0x10, 8)
data_ptr = struct.unpack("<Q", data_ptr_bytes)[0]
print(f"3. StorageImpl + 0x10          -> Data Buffer: {hex(data_ptr)}")

# Step 5.4: Read the first 4 floats at the raw buffer
raw_bytes = proc_layer.read(data_ptr, 16)
words = struct.unpack("<4I", raw_bytes)
floats = struct.unpack("<4f", raw_bytes)

print(f"\n4. Raw Values at {hex(data_ptr)}:")
print("   Hex Words: " + " ".join(f"0x{w:08x}" for w in words))
print("   Floats:    " + " ".join(f"{f:.8e}" for f in floats))
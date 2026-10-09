import torch
import torchvision
from torchvision.transforms import v2

# 1. Apply the exact normalization pipeline used during training
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 2. Load the CIFAR-10 test set
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

# CIFAR-10 class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 3. Extract the first batch of 4 test images and their ground-truth labels
images, labels = next(iter(test_loader))

print("[+] Ground-truth labels for test batch:")
for i in range(4):
    label_idx = labels[i].item()
    print(f"    Image {i}: label {label_idx} ({classes[label_idx]})")

# 4. Export the contiguous raw float32 buffer to a binary file
output_path = "models/test_batch.bin"
images.contiguous().numpy().tofile(output_path)
print(f"[+] Exported 4 test images to {output_path} ({images.numel() * 4} bytes)")
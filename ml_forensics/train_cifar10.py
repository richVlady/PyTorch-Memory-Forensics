import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split, ConcatDataset

# 1. Image preprocessing and normalization
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 2. Model Architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # Input: 32x32 -> Conv1(5x5): 28x28 -> Pool: 14x14
        # Conv2(5x5): 10x10 -> Pool: 5x5 -> 16 channels * 5 * 5 = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    device = torch.device("cpu")
    print(f"Executing on device: {device}")

    # 3. Download and merge CIFAR-10
    print("Loading CIFAR-10 dataset...")
    raw_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    raw_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([raw_train, raw_test])

    # 4. Strict 80 / 10 / 10 Split
    total_samples = len(full_dataset)                  # 60,000
    train_size = int(0.80 * total_samples)             # 48,000
    val_size = int(0.10 * total_samples)               # 6,000
    test_size = total_samples - train_size - val_size  # 6,000

    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Data partitioning: {len(train_set)} train | {len(val_set)} val | {len(test_set)} test")

    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 5. Initialize network, loss function, and optimizer
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 6. Training and validation loop
    epochs = 5
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f"[Epoch {epoch + 1}, Batch {i + 1:3d}/{len(train_loader)}] Loss: {running_loss / 200:.4f}")
                running_loss = 0.0

        # Validation step at end of epoch
        net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100.0 * val_correct / val_total
        print(f"--> Epoch {epoch + 1} Summary: Val Loss: {val_loss / len(val_loader):.4f} | Val Accuracy: {val_accuracy:.2f}%")

    print("Finished Training.")

    # 7. Final Test Set Evaluation
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * test_correct / test_total
    print(f"\nFinal Test Evaluation: {test_accuracy:.2f}% accuracy over {test_total} test images")

    # 8. Save Model Weights
    os.makedirs("models", exist_ok=True)
    weights_path = "models/cifar10_custom_cnn.pt"
    torch.save(net.state_dict(), weights_path)
    print(f"Model state_dict saved successfully to: {weights_path}")

if __name__ == '__main__':
    main()
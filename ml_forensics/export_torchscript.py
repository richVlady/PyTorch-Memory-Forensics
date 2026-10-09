import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Re-declare the identical network structure
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    weights_path = "models/cifar10_custom_cnn.pt"
    export_path = "models/cifar10_traced_model.pt"

    # 2. Instantiate and load trained weights
    model = Net()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}. Run training first.")
    
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    print("Loaded trained model weights successfully.")

    # 3. Create representative dummy input (Batch size: 1, Channels: 3, 32x32 image)
    example_input = torch.randn(1, 3, 32, 32)

    # 4. Trace the model to produce a TorchScript ScriptModule
    print("Tracing model execution graph...")
    traced_module = torch.jit.trace(model, example_input)

    # 5. Save the compiled TorchScript artifact
    traced_module.save(export_path)
    print(f"TorchScript model serialized to: {export_path}")

    # 6. Verify by loading and evaluating in TorchScript
    loaded_module = torch.jit.load(export_path)
    loaded_module.eval()
    
    with torch.no_grad():
        original_output = model(example_input)
        traced_output = loaded_module(example_input)

    # Check maximum numerical difference
    max_diff = torch.max(torch.abs(original_output - traced_output)).item()
    print(f"Max absolute discrepancy between Python and TorchScript: {max_diff:.8e}")
    assert max_diff < 1e-5, "Numerical deviation detected between models."
    print("Export verification complete. The model is ready for LibTorch C++ execution.")

if __name__ == "__main__":
    main()
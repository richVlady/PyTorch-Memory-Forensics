#include <torch/script.h> 
#include <torch/torch.h>  
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

int main(int argc, const char* argv[]) {
    // 1. Validate command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path-to-model> [path-to-test-batch.bin]\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string data_path = (argc >= 3) ? argv[2] : "../../models/test_batch.bin";

    std::cout << "[+] Loading TorchScript model from: " << model_path << "\n";

    // 2. Deserialize the model from disk
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "[-] Error loading the model:\n" << e.what() << "\n";
        return 1;
    }

    module.eval();
    std::cout << "[+] Model loaded successfully.\n";

    // 3. Load real test image data from binary file
    std::cout << "[+] Loading real CIFAR-10 test batch from: " << data_path << "\n";
    std::ifstream file(data_path, std::ios::binary);
    torch::Tensor input_tensor;

    if (file) {
        // Batch size 4, 3 channels, 32x32 pixels = 12,288 float elements
        const size_t num_elements = 4 * 3 * 32 * 32;
        std::vector<float> buffer(num_elements);
        file.read(reinterpret_cast<char*>(buffer.data()), num_elements * sizeof(float));

        // Wrap the raw memory buffer into a LibTorch Tensor and clone to own the heap memory
        input_tensor = torch::from_blob(buffer.data(), {4, 3, 32, 32}, torch::kFloat32).clone();
        std::cout << "[+] Successfully loaded real image tensor from disk.\n";
    } else {
        std::cerr << "[-] Warning: Could not open " << data_path << ". Falling back to random tensor.\n";
        input_tensor = torch::rand({4, 3, 32, 32}) * 2.0f - 1.0f;
    }

    // 4. Wrap inputs into an IValue vector
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // 5. Execute the forward pass
    std::cout << "[+] Executing forward pass...\n";

    // Explicit breakpoint target for GDB inspection:
    at::Tensor output = module.forward(inputs).toTensor();

    // 6. Inspect output predictions
    std::cout << "[+] Forward pass complete.\n";
    std::cout << "[+] Output tensor shape: " << output.sizes() << "\n";

    torch::Tensor predictions = torch::argmax(output, /*dim=*/1);

    const std::vector<std::string> classes = {
        "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    };

    std::cout << "\n[+] Inference Results on Real CIFAR-10 Images:\n";
    for (int i = 0; i < 4; ++i) {
        int pred_idx = predictions[i].item<int>();
        std::cout << "    Image " << i << ": predicted class " << pred_idx
                  << " (" << classes[pred_idx] << ")\n";
    }

    return 0;
}
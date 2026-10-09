import torch
import numpy as np
import os
import glob

print('--- Ground Truth Validation: Full Model Comparison ---')

# Load original model state_dict
try:
    state_dict = torch.load('models/cifar10_custom_cnn.pt', map_location='cpu', weights_only=True)
except Exception as e:
    print(f'[-] Error loading models/cifar10_custom_cnn.pt: {e}')
    exit(1)

# Get all extracted .npy files
extracted_files = glob.glob('output/pid31408.module.*.npy')
if not extracted_files:
    print('[-] No extracted .npy files found in output/')
    exit(1)

all_match = True
total_tensors = 0

print(f"{'Layer Name':<25} | {'Original Shape':<18} | {'Extracted Shape':<18} | {'Match'}")
print("-" * 80)

for filepath in sorted(extracted_files):
    filename = os.path.basename(filepath)
    # Filename format: pid31408.module.conv1.weight.npy
    # Map to PyTorch key: conv1.weight
    parts = filename.replace('.npy', '').split('.')
    if len(parts) >= 3:
        layer_name = f"{parts[-2]}.{parts[-1]}"
    else:
        continue
        
    # Find matching key in state_dict (ignoring prefix differences)
    matching_key = None
    for k in state_dict.keys():
        if k.endswith(layer_name):
            matching_key = k
            break
            
    if not matching_key:
        print(f"{layer_name:<25} | {'Not Found':<18} | {'-':<18} | {'N/A'}")
        all_match = False
        continue

    original_weights = state_dict[matching_key].numpy()
    extracted_weights = np.load(filepath)
    
    is_exact_match = np.array_equal(original_weights, extracted_weights)
    if not is_exact_match:
        all_match = False
        
    orig_shape_str = str(original_weights.shape)
    ext_shape_str = str(extracted_weights.shape)
    match_str = "True" if is_exact_match else "FALSE"
    
    print(f"{layer_name:<25} | {orig_shape_str:<18} | {ext_shape_str:<18} | {match_str}")
    total_tensors += 1

print("-" * 80)
if all_match:
    print(f"[+] SUCCESS: All {total_tensors} extracted tensors perfectly match the original model weights!")
else:
    print("[-] FAILURE: Mismatches detected in the extracted tensors.")
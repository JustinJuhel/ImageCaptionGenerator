import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU.")
    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    # Get the name of the GPU
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. PyTorch will use the CPU.")

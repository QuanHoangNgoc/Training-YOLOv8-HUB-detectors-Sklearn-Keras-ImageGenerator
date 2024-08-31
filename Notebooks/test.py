import tensorflow as tf 
print(tf.__version__) 

import torch 
print(torch.__version__)  # Should show the PyTorch version
print(torch.version.cuda)  # Should show the CUDA version
print(torch.cuda.is_available())  # Should return True if the GPU is available

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available")


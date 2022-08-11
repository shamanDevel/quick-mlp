import os
import torch
import torch.nn.functional
import torch.autograd
import torch.nn.init
from typing import Optional
import imageio
import matplotlib.pyplot as plt

torch.classes.load_library(os.path.join(os.path.split(__file__)[0], "../bin/qmlp.so"))
print(torch.classes.loaded_libraries)

torch.classes.qmlp.QuickMLP.set_debug_mode(True)
pullpush = torch.classes.qmlp.utils.pullpush

# load test image from the "Painter" dataset
ground_truth_image = imageio.imread("340.jpg")
print(ground_truth_image.shape)
print(ground_truth_image.dtype)
plt.figure()
plt.imshow(ground_truth_image)
plt.title("Input ground truth")
plt.show()

# to PyTorch and to B,C,H,W format
device = torch.device("cuda")
ground_truth_gpu = torch.from_numpy(ground_truth_image).to(dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0) / 255.0
B, C, H, W = ground_truth_gpu.shape

# create sparse mask with 5% of pixels set
mask = (torch.rand((B, H, W), dtype=torch.float32, device=device) > 0.95) * 1.0

# inpainting
inpainted_image = pullpush(mask, ground_truth_gpu)
# and show
fig, axes = plt.subplots(1, 2)
axes[0].imshow(mask[0].cpu().numpy())
axes[0].set_title("Mask")
axes[1].imshow(inpainted_image[0].permute(1,2,0).cpu().numpy())
axes[1].set_title("Inpainted")
plt.show()
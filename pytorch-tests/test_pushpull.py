import os
import numpy as np
import torch
import torch.nn.functional
import torch.autograd
import torch.nn.init
from typing import Optional
import imageio
import matplotlib.pyplot as plt
import torch.optim

from import_library import load_library
load_library()

# fetch pull-push implementation
pullpush = torch.classes.qmlp.utils.pullpush

# load test image from the "Painter" dataset
ground_truth_image = imageio.imread("pushpull_input.jpg")
print(ground_truth_image.shape)
print(ground_truth_image.dtype)
#plt.figure()
#plt.imshow(ground_truth_image)
#plt.title("Input ground truth")
#plt.show()

#prepare output
os.makedirs("output", exist_ok=True)
def save(img, name):
    if len(img.shape)==3: #B,H,W
        img = torch.stack([img,img,img], dim=1) # B,3,H,W
    img = img[0] # first batch
    img8 = np.clip(img.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    imageio.imwrite(os.path.join("output", name), img8)

# to PyTorch and to B,C,H,W format
device = torch.device("cuda")
ground_truth_gpu = torch.from_numpy(ground_truth_image).to(dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0) / 255.0
B, C, H, W = ground_truth_gpu.shape

# create sparse mask with 5% of pixels set
mask = (torch.rand((B, H, W), dtype=torch.float32, device=device) > 0.95) * 1.0
mask2 =  mask.unsqueeze(1)
save(mask, "pullpush-mask.png")

# inpainting
inpainted_image = pullpush(mask, ground_truth_gpu)
save(inpainted_image, "pullpush-inpainted.png")
# and show
#fig, axes = plt.subplots(1, 2)
#axes[0].imshow(mask[0].cpu().numpy())
#axes[0].set_title("Mask")
#axes[1].imshow(inpainted_image[0].permute(1,2,0).cpu().numpy())
#axes[1].set_title("Inpainted")
#plt.show()

# train input
data_in = torch.rand_like(ground_truth_gpu)
data_in.requires_grad_(True)
adam = torch.optim.Adam([data_in], lr=0.2)
for i in range(10):
    adam.zero_grad()
    out = pullpush(mask, data_in)
    data_masked = (data_in * mask2).detach()
    data_inpainted = out.detach()
    loss = torch.nn.functional.mse_loss(out, ground_truth_gpu)
    loss.backward()
    adam.step()
    save(data_masked, "pullpush-optim-DataMasked%02d.png"%i)
    save(data_inpainted, "pullpush-optim-DataInpainted%02d.png" % i)
    print("Optimize for the input data, loss:", loss.item())

# train mask
mask_raw_in = torch.randn_like(mask)
mask_raw_in.requires_grad_(True)
adam = torch.optim.Adam([mask_raw_in], lr=0.2)
probability = 0.05 # 5% of pixels
min_value = 1e-5
for i in range(10):
    adam.zero_grad()

    # transform and normalize mask
    importance1 = torch.nn.functional.softplus(mask_raw_in)
    m = torch.mean(importance1,dim=[1, 2], keepdim=True)
    m = torch.clamp(m, min=1e-7)
    importance2 = min_value + importance1 * ((probability - min_value) / m)

    out = pullpush(importance2, ground_truth_gpu)
    data_inpainted = out.detach()
    loss = torch.nn.functional.mse_loss(out, ground_truth_gpu)
    loss.backward()
    adam.step()
    save(importance2.detach(), "pullpush-optim-MaskRaw%02d.png"%i)
    save(data_inpainted, "pullpush-optim-MaskInpainted%02d.png" % i)
    print("Optimize for the mask, loss:", loss.item())

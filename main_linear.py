# %%
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from tqdm import trange
import os
import imageio.v2 as imageio
import cv2
from deconv_cppn import CPPN
from deconv_linear import PPN
from torch import Tensor
import torch.optim as optim
import torch

from piq.ms_ssim import multi_scale_ssim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(0)
res = 512
# Download image, take a square crop from the center
# image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg' # fox
image_url = 'https://upload.wikimedia.org/wikipedia/commons/1/1e/Sunrise_over_the_sea.jpg' # sunrise
img = imageio.imread(image_url)[..., :3] / 255.
if min(img.shape[:2]) < res:
    img = cv2.resize(img, (res, res))

c = [img.shape[0]//2, img.shape[1]//2]
r = res//2
img = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r]

# Fourier feature mapping
def input_mapping(x, B):
  if B is None:
    return x
  else:
    x_proj = (2.*np.pi*x) @ B.T
    return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)

mapping_size = 128
scale = 10.0
B_gauss = np.random.randn(mapping_size, 2)
B_gauss = B_gauss * scale
train_downscales = 2

num_parents = 1
num_children = 1
elitism = 1
evo_iters = 100
sgd_iters = 10000
LR = .0001




coords = np.linspace(0, 1, img.shape[0], endpoint=False)
x_test = np.stack(np.meshgrid(coords, coords), -1)
train_downscale = 2
test_data = [x_test, img]
train_data = [x_test[::2**train_downscales, ::2**train_downscales],
              img[::2**train_downscales, ::2**train_downscales]]

X = input_mapping(train_data[0], B_gauss)
X = X.transpose(2, 0, 1)
print("input shape", X.shape)

img = test_data[1].transpose(2, 0, 1)
# img = train_data[1].transpose(2, 0, 1)
target = Tensor(img.astype(np.float32)).to(DEVICE)
print("target shape", target.shape)
X = Tensor(np.array(X.astype(np.float32))).to(DEVICE)

num_devices = 2
targs = [target.clone().to(f"cuda:{i}") for i in range(num_devices)]
Xs = [X.clone().to(f"cuda:{i}") for i in range(num_devices)]
device_targets = {
    torch.device(f"cuda:{i}"): targs[i] for i in range(num_devices)
}
device_Xs = {
    f"cuda:{i}": Xs[i] for i in range(num_devices)
}



evo_pbar = trange(evo_iters)

def loss_fn(Y):
    t = device_targets[Y.device]
    return torch.mean((Y - t)**2)
    # x = target.unsqueeze(0)
    # y = Y.unsqueeze(0)
    # return torch.mean((Y - target)**2) + multi_scale_ssim(x,y, data_range=1.0)
    # return multi_scale_ssim(x,y, data_range=1.0)

def reproduce(nets):
    children = nets[:elitism] 
    while len(children) < num_children:
        rand_parent = nets[np.random.randint(0, len(nets))]
        child = rand_parent.clone()
        child.mutate()
        child.mutate()
        child.mutate()
        children.append(child)
    return children

def trunc_select(children):
    children = sorted(children, key=lambda x: x.loss)
    parents = children[:num_parents]
    return parents

tourn_size = 3
tourn_winners = 1
def tourn_select(children):
    winners = []
    while len(winners) < num_parents:
        rand_children_idxs = np.random.choice(len(children), tourn_size, replace=False)
        rand_children = sorted([children[i] for i in rand_children_idxs], key=lambda x: x.loss)
        winners.extend(rand_children[:tourn_winners])
    return winners
        
def select(children):
    return trunc_select(children)
    # return tourn_select(children)



convs = [(-1,3)]
strides = [2] 

#%%
# show example network:
# CPPN(
#         mapping_size*2, 12, 12, 0.80, 
#         n_upsamples=train_downscales + int(sum(np.log2(strides))),
#         convs=convs,
#         strides = strides,
#         device='cuda',
#         ).draw_nx()
#%%

def make_deconv_cppn(device):
    return CPPN(
        mapping_size*2, 3, 3, 0.80, 
        n_upsamples=train_downscales + int(sum(np.log2(strides))),
        convs=convs,
        strides = strides,
        device=device,
        )
def make_ppn(device):
    return PPN(mapping_size*2, 3, 3, 0.80, 
        n_upsamples=train_downscales + int(sum(np.log2(strides))),
        convs=convs,
        strides = strides,
        device=device)

def make_cppn(device):
    return CPPN(mapping_size*2, 32, 3, 0.80, 
        n_upsamples=0,
        convs=None,
        strides = None,
        device=device)

parents = [
    #  make_deconv_cppn(f"cuda:" + str(i % num_devices)) for i in range(num_parents)
     make_ppn(f"cuda:" + str(i % num_devices)) for i in range(num_parents)
    #  make_cppn(f"cuda:" + str(i % num_devices)) for i in range(num_parents)
]

best = parents[0]
all_losses = []
best_losses = []
try:
    for e in evo_pbar:
        children = reproduce(parents)
        sgd_pbar = trange(sgd_iters, leave=False)
        all_params = []
        for i, child in enumerate(children):
            child.to(f"cuda:" + str(i % num_devices))
            all_params += list(child.params)
        optimizer = optim.Adam(all_params, lr=LR)
        for k in sgd_pbar:
            Ys = [net(device_Xs[net.device]) for net in children]
            child_losses = [loss_fn(y).to(DEVICE) for y in Ys]
            for i, child in enumerate(children):
                child.loss = child_losses[i].item()
            loss = torch.stack(child_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ms_per_iter = 1000 * (sgd_pbar.last_print_t - sgd_pbar.start_t) / (k + 1)
            sgd_pbar.set_description(
                f"loss: {loss.detach().cpu().numpy().mean():.4f}, {ms_per_iter:.1f} ms/iter")
            all_losses.append(loss.item())
            best_losses.append(best.loss)
        parents = select(children)
        
        if parents[0].loss < best.loss:
            best = parents[0].clone(new_id=False)
        evo_pbar.set_description(f"loss: {best.loss:.4f} best: [id:{best.id} params:{best.num_params}]")
        
except KeyboardInterrupt:
    if parents[0].loss < best.loss:
            best = parents[0].clone(new_id=False)
    pass       
plt.plot(all_losses, alpha=.5, label="mean")
plt.plot(best_losses, label="best")
plt.legend()
plt.show()

output = best(X)

print("final num params:", best.num_params)
p = best.num_params_by_module
print("num params:", "cppn:", p[0], 'conv:', p[1], 'deconv:', p[2], 'total:', best.num_params)
print("final loss:", loss_fn(output).detach().cpu().numpy().mean())

best.draw_nx()

plt.imshow(np.stack([y.detach().cpu().numpy() for y in output], axis=-1))
plt.show()

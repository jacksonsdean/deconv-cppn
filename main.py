# %%
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from tqdm import trange
import os
import imageio.v2 as imageio
import cv2
from deconv_cppn import CPPN
from torch import Tensor
import torch.optim as optim
import torch

from piq.ms_ssim import multi_scale_ssim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(0)
res = 512
# Download image, take a square crop from the center
image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
# image_url = 'https://upload.wikimedia.org/wikipedia/commons/1/1e/Sunrise_over_the_sea.jpg'
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

mapping_size = 256
scale = 10.0
B_gauss = np.random.randn(mapping_size, 2)
B_gauss = B_gauss * scale

num_downscales = 1

coords = np.linspace(0, 1, img.shape[0], endpoint=False)
x_test = np.stack(np.meshgrid(coords, coords), -1)
train_downscale = 8
test_data = [x_test, img]
train_data = [x_test[::2**num_downscales, ::2**num_downscales],
              img[::2**num_downscales, ::2**num_downscales]]

X = input_mapping(train_data[0], B_gauss)
X = X.transpose(2, 0, 1)
print("input shape", X.shape)

img = test_data[1].transpose(2, 0, 1)
target = Tensor(img.astype(np.float32)).to(DEVICE)
print("target shape", target.shape)
X = Tensor(np.array(X.astype(np.float32))).to(DEVICE)


num_parents = 2
num_children = 10
elitism = 1
evo_iters = 100
sgd_iters = 100
LR = .03

evo_pbar = trange(evo_iters)

def loss_fn(Y):
    x = target.unsqueeze(0)
    y = Y.unsqueeze(0)
    return torch.mean((Y - target)**2)
    # return torch.mean((Y - target)**2) + multi_scale_ssim(x,y, data_range=1.0)
    # return multi_scale_ssim(x,y, data_range=1.0)

def reproduce(nets):
    children = nets[:elitism] 
    while len(children) < num_children:
        rand_parent = nets[np.random.randint(0, len(nets))]
        child = rand_parent.clone()
        child.mutate()
        children.append(child)
    return children

def trunc_select(children):
    children = [(child, loss_fn(child(X))) for child in children]
    children = sorted(children, key=lambda x: x[1])
    parents = children[:num_parents]
    return [p[0] for p in parents]

tourn_size = 3
tourn_winners = 1
def tourn_select(children):
    children = [(child, loss_fn(child(X)).item()) for child in children]
    winners = []
    while len(winners) < num_parents:
        rand_children_idxs = np.random.choice(len(children), tourn_size, replace=False)
        rand_children = sorted([children[i] for i in rand_children_idxs], key=lambda x: x[1])
        winners.extend(rand_children[:tourn_winners])
    return [p[0] for p in winners]
        
def select(children):
    # return trunc_select(children)
    return tourn_select(children)

convs = [(-1, 3)]
strides = [1]

losses = []
parents = [
    CPPN(
        mapping_size*2, 0, 32, 0.80, 
        n_deconvs=num_downscales+(sum(strides)-len(strides)), 
        convs=convs,
        strides = strides,
        ) for _ in range(num_parents)
]
try:
    for e in evo_pbar:
        children = reproduce(parents)
        sgd_pbar = trange(sgd_iters, leave=False)
        all_params = []
        for child in children:
            all_params += list(child.params)
        optimizer = optim.Adam(all_params, lr=LR)
        for k in sgd_pbar:
            Ys = torch.stack([net(X) for net in children])
            loss = torch.stack([loss_fn(y) for y in Ys]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ms_per_iter = 1000 * (sgd_pbar.last_print_t - sgd_pbar.start_t) / (k + 1)
            sgd_pbar.set_description(
                f"loss: {loss.detach().cpu().numpy().mean():.4f}, {ms_per_iter:.1f} ms/iter")
            losses.append(loss.item())
        parents = select(children)
        evo_pbar.set_description(f"loss: {losses[-1]:.4f} top_id: {parents[0].id} num_params: {parents[0].num_params}")
except KeyboardInterrupt:
    children = [(child, loss_fn(child(X))) for child in children]
    children = sorted(children, key=lambda x: x[1])
    parent = children[:num_parents]
    pass    
plt.plot(losses)
plt.show()

output = parents[0](X)

print("final num params:", parents[0].num_params)
p = parents[0].num_params_by_module
print("num params:", "cppn:", p[0], 'conv:', p[1], 'deconv:', p[2], 'total:', parents[0].num_params)
print("final loss:", loss_fn(output).detach().cpu().numpy().mean())

# parent.draw_nx()

plt.imshow(np.stack([y.detach().cpu().numpy() for y in output], axis=-1))
plt.show()

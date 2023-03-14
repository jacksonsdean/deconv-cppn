import torch

def tanh(x):
  return torch.tanh(x)

def sigmoid(x):
  return torch.sigmoid(x)

def triangle(x):
  return 1 - 2 *torch.arccos((1 - .0001) * torch.sin(2 * torch.pi * x))/torch.pi

def square(x):
  return 2* torch.arctan(torch.sin(2 *torch.pi* x)/.0001)/torch.pi

def sawtooth(x):
  return (1 + triangle((2*x - 1)/4.0) * square(x/2.0)) / 2.0

def gaussian(x):
  return torch.exp(-x**2)

def relu(x):
  return torch.relu(x)

def identity(x):
  return x

def sin(x):
  return torch.sin(x)

def cos(x):
  return torch.cos(x)

# all_activations = [tanh, sigmoid, triangle, square, sawtooth, gaussian, relu, identity]
all_activations = [sigmoid, gaussian, relu, identity, sin, cos]
# all_activations = [sigmoid]
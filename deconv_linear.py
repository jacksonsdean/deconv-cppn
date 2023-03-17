import copy
import torch
import numpy as np
import enum
from torch import Tensor
from graph_util import feed_forward_layers, get_incoming_connections, get_outgoing_connections, is_valid_connection
from networkx.drawing.nx_agraph import graphviz_layout
from torch_activations import all_activations
from torch.nn import ConvTranspose2d, Conv2d, Sequential, Upsample, ReflectionPad2d

def upscale_conv2d(in_channels, out_channels, kernel_size, stride, padding, device="cpu"):
    # return ConvTranspose2d(in_channels,out_channels,kernel_size, stride=stride, padding=padding, output_padding=1,device=device)
    layer = Sequential(
        Upsample(scale_factor=2, mode='bilinear'),
        ReflectionPad2d(1),
        Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0, device=device)
   )
    layer.__dict__['kernel_size'] = kernel_size
    layer.__dict__['in_channels'] = in_channels
    return layer

class PPNLayer(torch.nn.Module):
    def __init__(self, in_nodes, out_nodes, device="cpu"):
        super(PPNLayer, self).__init__()
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.device = device
        self.weight = torch.nn.Parameter(torch.randn(out_nodes, device=device))
        self.bias = torch.nn.Parameter(torch.randn((1,out_nodes), device=device))
        
    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(self.in_nodes, -1)
        out = (x.mT * self.weight + self.bias).mT
        out = out.reshape(x_shape)
        return out

class PPN():
    max_id = -1
    @staticmethod
    def get_id():
        PPN.max_id += 1
        return PPN.max_id
    
    def __init__(self, n_inputs, n_hidden=0, n_outputs=6, in_shape=(64,64), device="cpu", n_upsamples=3, convs=[(-1, 3)], strides=[2]):
        self.device = device
        self.id = PPN.get_id()
        self.conv_layers = []
        if convs is not None and len(convs) > 0:
            for i in range(len(convs)):
                in_dims = convs[i][0]
                if in_dims == -1:
                    in_dims = n_inputs
                out_dims = convs[i][1]
                self.conv_layers.append(Conv2d(in_dims,out_dims,3, stride=strides[i], padding=1,device=device))
            n_inputs = out_dims
        self.linear_layers = []
        self.linear_layers.append(PPNLayer(n_inputs, n_hidden, device=device))
        self.linear_layers.append(PPNLayer(n_hidden, n_outputs, device=device))
        self.layers = None
        self.deconv_layers = []
        self.loss = torch.inf
        if n_upsamples > 0:
            self.deconv_layers.append(upscale_conv2d(n_outputs,3,3, stride=2, padding=1, device=device))
            for _ in range(n_upsamples-1):
                self.deconv_layers.append(upscale_conv2d(3,3,3, stride=2, padding=1, device=device))
        
    @property
    def params(self):
        return [p for l in self.deconv_layers for p in l.parameters()] + [p for l in self.conv_layers for p in l.parameters()] + [p for l in self.linear_layers for p in l.parameters()]
    @property
    def num_params_by_module(self):
        conv_params = 0
        for layer in self.conv_layers:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    conv_params += param.numel()
                    
        linear_params = 0
        for layer in self.linear_layers:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    linear_params += param.numel()
        
        deconv_params = 0
        for layer in self.deconv_layers:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    deconv_params += param.numel()
        return conv_params , linear_params, deconv_params
    @property
    def num_params(self):
        return sum(self.num_params_by_module)

    def mutate(self, 
            **kwargs):
        return
        

    def to(self, device):
        self.device = device
        for i, l in enumerate(self.conv_layers):
            self.conv_layers[i] = l.to(device)
        for i, l in enumerate(self.deconv_layers):
            self.deconv_layers[i] = l.to(device)
        for i, l in enumerate(self.linear_layers):
            self.linear_layers[i] = l.to(device)
    
    def forward(self, inputs):
        inputs = inputs.to(self.device)
        for conv_layer in self.conv_layers:
            inputs = conv_layer(inputs)
        for lin_layer in self.linear_layers:
            inputs = lin_layer(inputs)
            inputs = torch.sigmoid(inputs)
        for deconv_layer in self.deconv_layers:
            inputs = inputs.unsqueeze(0)
            inputs = deconv_layer(inputs)
            inputs = inputs.squeeze(0)
        inputs = torch.sigmoid(inputs)
        return inputs
    
    def to_nx(self):
        import networkx as nx
        G = nx.DiGraph()
        for i, layer in enumerate(self.conv_layers):
            G.add_node(f'CONV {i}\n{layer.kernel_size}x{layer.in_channels}', type='conv')
        for i, layer in enumerate(self.deconv_layers):
            G.add_node(f'DECONV {i}\n{layer.kernel_size}x{layer.in_channels}', type='deconv')
            
        for i in range(len(self.conv_layers)-1):
            G.add_edge(f'CONV {i}\n{self.conv_layers[i].kernel_size}x{self.conv_layers[i].in_channels}', f'CONV {i+1}\n{self.conv_layers[i+1].kernel_size}x{self.conv_layers[i+1].in_channels}')
        for i in range(len(self.deconv_layers)-1):
            G.add_edge(f'DECONV {i}\n{self.deconv_layers[i].kernel_size}x{self.deconv_layers[i].in_channels}', f'DECONV {i+1}\n{self.deconv_layers[i+1].kernel_size}x{self.deconv_layers[i+1].in_channels}')
        
        for i, layer in enumerate(self.linear_layers):
            G.add_node(f'LIN {i}\n{layer.in_nodes}x{layer.out_nodes}', type='lin')
        for i in range(len(self.linear_layers)-1):
            G.add_edge(f'LIN {i}\n{self.linear_layers[i].in_nodes}x{self.linear_layers[i].out_nodes}', f'LIN {i+1}\n{self.linear_layers[i+1].in_nodes}x{self.linear_layers[i+1].out_nodes}')
        for i in range(len(self.conv_layers)):
            G.add_edge(f'CONV {i}\n{self.conv_layers[i].kernel_size}x{self.conv_layers[i].in_channels}', f'LIN 0\n{self.conv_layers[i].out_channels}x{self.linear_layers[0].in_nodes}')
            
        
        return G
    
    def draw_nx(self):
        import matplotlib.pyplot as plt
        import networkx as nx
        fig = plt.figure(figsize=(10,30))
        G = self.to_nx()
        pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")
        nx.draw(
            G,
            with_labels=True,
            pos=pos,
            labels={n:n for n in G.nodes(data=False)},
            node_size=800,
            font_size=6,
            node_shape='s',
            node_color=['lightsteelblue' if 'LIN' in n  else 'lightgreen' for n in G.nodes()  ]
            )
        plt.annotate('# params: ' + str(self.num_params), xy=(1.0, 1.0), xycoords='axes fraction', fontsize=12, ha='right', va='top')
        plt.show()
                    
    def __call__(self, X):
        return self.forward(X)
    
    def clone(self, new_id=True):
        """ Create a copy of this genome. """
        id = self.id if (not new_id) else type(self).get_id()
        child = copy.deepcopy(self)
        child.id = id
        return child
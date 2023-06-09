import copy
import torch
import numpy as np
import enum
from torch import Tensor
from graph_util import feed_forward_layers, get_incoming_connections, get_outgoing_connections, is_valid_connection
from networkx.drawing.nx_agraph import graphviz_layout
from torch_activations import all_activations
from torch.nn import ConvTranspose2d, Conv2d, Sequential, Upsample, ReflectionPad2d,MaxPool2d   


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


class NodeType(enum.Enum):
    INPUT  = 0
    OUTPUT = 1
    HIDDEN = 3

class Node():
    def __init__(self, id, type:NodeType, fn:callable,layer=None,device="cpu"):
        self.id = id
        self.type = type
        self.value = 0
        self.connections = {}
        self.fn = fn
        self.value = None
        self.bias = Tensor([0.0]).to(device)
        self.incoming = None
        self.layer = layer
        

class Connection():
    def __init__(self, in_node, out_node, weight, device="cpu"):
        self.id = (in_node.id, out_node.id)
        self.weight = Tensor([weight]).to(device)
        self.weight.requires_grad = True
        self.enabled = True

class CPPN():
    max_id = -1
    @staticmethod
    def get_id():
        CPPN.max_id += 1
        return CPPN.max_id
    
    @staticmethod
    def get_constant_inputs(res, coord_range=(-0.5, 0.5), add_bias=False):
        all_vals = []
        for r in res:
            vals = np.linspace(coord_range[0], coord_range[1], r)
            all_vals.append(vals)
        all_vals = np.meshgrid(*all_vals)
        if add_bias:
            all_vals.append(np.ones(all_vals[0].shape))
        return np.stack(all_vals, axis=0)
    
    def __init__(self, n_inputs, n_hidden=0, n_outputs=6, init_connection_prob=0.8, device="cpu", n_upsamples=3, convs=[(-1, 3)], strides=[2], pools=None, without_conv=False):
        self.device = torch.device(device)
        self.id = CPPN.get_id()
        self.conv_layers = []
        assert pools is None or len(pools) == len(convs), "pools must be same length as convs for now"
        if convs is not None and len(convs) > 0:
            for i in range(len(convs)):
                in_dims = convs[i][0]
                if in_dims == -1:
                    in_dims = n_inputs
                out_dims = convs[i][1]
                self.conv_layers.append(Conv2d(in_dims,out_dims,3, stride=strides[i], padding=1,device=device))
                if pools is not None:
                    self.conv_layers.append(MaxPool2d(pools[i]))
                    self.conv_layers[-1].__dict__['in_channels'] = f' POOL{pools[i]}'
            n_inputs = out_dims
        self.hidden_dim = n_inputs
        self.init_genome(n_inputs, n_hidden, n_outputs, init_connection_prob)
        self.layers = None
        self.deconv_layers = []
        self.loss = torch.inf
        if without_conv:
            for _ in range(n_upsamples):
                self.deconv_layers.append(Upsample(scale_factor=2, mode='bilinear'))
                self.deconv_layers[-1].__dict__['kernel_size'] = ' '
                self.deconv_layers[-1].__dict__['in_channels'] = 2
        else:
            if n_upsamples > 0:
                self.deconv_layers.append(upscale_conv2d(n_outputs,3,3, stride=2, padding=1, device=device))
                for _ in range(n_upsamples-1):
                    self.deconv_layers.append(upscale_conv2d(3,3,3, stride=2, padding=1, device=device))
            
    def forward(self, inputs):
        inputs = inputs.to(self.device)
        for conv_layer in self.conv_layers:
            inputs = conv_layer(inputs)
        
        self.reset_activations(inputs)
        
        for idx, input_node in enumerate(self.input_nodes):
            input_node.value = inputs[idx]
            # input_node.value = input_node.fn(input_node.value)

        if self.layers is None: 
            self.layers = feed_forward_layers(self)
            
        for layer in self.layers:
            for node_id in layer:
                node = self.nodes[node_id]
                if node.incoming is None:
                    node.incoming = get_incoming_connections(self, node)
                for cxn in node.incoming:
                    node.value = node.value + self.nodes[cxn.id[0]].value * cxn.weight  + node.bias
                node.value = node.fn(node.value)
                
        sorted_outputs = sorted(self.output_nodes, key=lambda n: n.id)
        latent = torch.stack([s.value for s in sorted_outputs])
        for i in range(len(self.deconv_layers)):
            latent = latent.unsqueeze(0)
            latent = self.deconv_layers[i](latent)
            latent = latent.squeeze(0)
        latent = torch.sigmoid(latent)
        return latent
        
    def init_genome(self, n_inputs, n_hidden, n_outputs, init_connection_prob):
        self.nodes = {}
        self.connections = {}
        self.init_nodes(n_inputs, n_outputs, n_hidden)
        self.init_connections(init_connection_prob)
        self.update_node_layers()

    def init_nodes(self, n_inputs, n_outputs, n_hidden):
        for i in range(n_inputs):
            id = -i-1
            self.nodes[id] = Node(id, NodeType.INPUT, self.random_activation(), layer=0,device=self.device)
        for i in range(n_outputs):
            id = -n_inputs-i-1
            self.nodes[id] = Node(id, NodeType.OUTPUT, self.random_activation(), layer=2,device=self.device)
        for i in range(n_hidden):
            self.nodes[i] = Node(i, NodeType.HIDDEN, self.random_activation(), layer=1, device=self.device)
        self.update_node_layers()

    def init_connections(self,c_prob):
        this_layer = self.input_nodes
        next_layer = self.output_nodes if len(self.hidden_nodes) == 0 else self.hidden_nodes
        while next_layer is not None:
            for n in this_layer:
                for o in next_layer:
                    r = self.random_uniform(minval=0, maxval=1)
                    if next_layer == self.output_nodes:
                        r = 0.0
                    if r < c_prob:
                        self.add_connection(n, o)
            this_layer = next_layer
            next_layer = None if this_layer==self.output_nodes else self.output_nodes
        self.update_node_layers()

    
    def update_node_layers(self):
        for n in self.nodes.values():
            n.incoming=None
        self.layers = feed_forward_layers(self)
        for node in self.input_nodes:
            node.layer = 0
        for layer_index, layer in enumerate(self.layers):
            for node_id in layer:
                self.nodes[node_id].layer = layer_index + 1
        for node in self.nodes.values():
            node.incoming = get_incoming_connections(self, node)
        
    def random_node(self):
        idx = np.random.randint(0, len(self.nodes))
        return self.nodes[list(self.nodes.keys())[idx]]
    
    def random_connection(self):
        idx = np.random.randint(0, len(self.connections))
        return self.connections[list(self.connections.keys())[idx]]
    
    def add_connection(self, node1=None, node2=None, weight=None):
        self.update_node_layers()
        for _ in range(20):
            if node1 is None:
                node1 = self.random_node()
            if node2 is None:
                node2 = self.random_node()
            if not is_valid_connection(self.nodes, (node1.id, node2.id)):
                continue
            c = Connection(node1, node2, self.random_uniform() if weight is None else weight,device=self.device)
            self.connections[c.id] = c
            self.update_node_layers()
            break
                

    def disable_connection(self):
        self.random_connection().enabled =False
        self.update_node_layers()

    def add_node(self):
        c = self.random_connection()
        if not c.enabled:
            return
        c.enabled = False
        new_node = Node(max([n for n in self.nodes.keys()])+1, NodeType.HIDDEN, self.random_activation(),device=self.device)
        self.nodes[new_node.id] = new_node
        self.add_connection(self.nodes[c.id[0]], new_node, weight=1.0)
        self.add_connection(new_node, self.nodes[c.id[1]])
        self.update_node_layers()

    def remove_node(self):
        n = self.random_node()
        if n.type == NodeType.INPUT or n.type == NodeType.OUTPUT:
            return
        incoming = get_incoming_connections(self, n)
        outgoing = get_outgoing_connections(self, n)
        for cxn in incoming:
            del self.connections[cxn.id]
        for cxn in outgoing:
            del self.connections[cxn.id]
        del self.nodes[n.id]
        self.update_node_layers()

    @property
    def params(self):
        return [c.weight for c in self.enabled_connections] + [n.bias for n in self.hidden_nodes] + [p for l in self.deconv_layers for p in l.parameters()] + [p for l in self.conv_layers for p in l.parameters()]
    @property
    def num_params_by_module(self):
        cppn_params = len(self.params) - len(self.deconv_layers) - len(self.conv_layers)
        conv_params = 0
        for layer in self.conv_layers:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    conv_params += param.numel()
        deconv_params = 0
        for layer in self.deconv_layers:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    deconv_params += param.numel()
        return cppn_params , conv_params , deconv_params
    @property
    def num_params(self):
        return sum(self.num_params_by_module)
    @property
    def input_nodes(self):
        return [n for n in self.nodes.values() if n.type == NodeType.INPUT]
    @property
    def output_nodes(self):
        return [n for n in self.nodes.values() if n.type == NodeType.OUTPUT]
    @property
    def hidden_nodes(self):
        return [n for n in self.nodes.values() if n.type == NodeType.HIDDEN]
    @property
    def enabled_connections(self):
        return [c for c in self.connections.values() if c.enabled]
    
    def mutate_weights(self, prob):
        for _, c in self.connections.items():
            r = np.random.uniform(0,1)
            if r < prob:
                c.weight = c.weight.detach() + self.random_normal()
        for n in self.nodes.values():
            n.incoming=None
            self.layers=None
                
    def mutate_activations(self, prob):
        for _, n in self.nodes.items():
            r = np.random.uniform(0,1)
            if r < prob:
                n.fn = self.random_activation()
        self.update_node_layers()

    def mutate(self, 
            add_node = 0.8,
            add_connection = 0.8,
            remove_node = 0.1,
            disable_connection = 0.1,
            mutate_activations = .1,
            mutate_weights = 0.01):
        
        if self.random_uniform(0.0,1.0) < add_node:
            self.add_node()
        if self.random_uniform(0.0,1.0) < remove_node:
            self.remove_node()
        if self.random_uniform(0.0,1.0) < add_connection:
            self.add_connection()
        if self.random_uniform(0.0,1.0) < disable_connection:
            self.disable_connection()
            
        self.mutate_activations(mutate_activations)
        self.mutate_weights(mutate_weights)
        self.update_node_layers()

    def to_nx(self):
        import networkx as nx
        G = nx.DiGraph()
        for i, layer in enumerate(self.conv_layers):
            G.add_node(f'CONV {i}\n{layer.kernel_size}x{layer.in_channels}', type='conv')
        for i, layer in enumerate(self.deconv_layers):
            G.add_node(f'UPSAMPLE {i}\n{layer.kernel_size}x{layer.in_channels}', type='deconv')
            
        for i in range(len(self.conv_layers)-1):
            G.add_edge(f'CONV {i}\n{self.conv_layers[i].kernel_size}x{self.conv_layers[i].in_channels}', f'CONV {i+1}\n{self.conv_layers[i+1].kernel_size}x{self.conv_layers[i+1].in_channels}')
        for i in range(len(self.deconv_layers)-1):
            G.add_edge(f'UPSAMPLE {i}\n{self.deconv_layers[i].kernel_size}x{self.deconv_layers[i].in_channels}', f'UPSAMPLE {i+1}\n{self.deconv_layers[i+1].kernel_size}x{self.deconv_layers[i+1].in_channels}')
        
        for n in self.nodes.values():
            G.add_node(n.id, type=n.type.name, fn=n.fn)
        for c in self.connections.values():
            if c.enabled:
                G.add_edge(c.id[0], c.id[1], weight=c.weight.item())
        if len(self.conv_layers) > 0:
            for n in self.input_nodes:
                G.add_edge(f"CONV {len(self.conv_layers)-1}\n{self.conv_layers[-1].kernel_size}x{self.conv_layers[-1].in_channels}", n.id)
        for n in self.output_nodes:
            G.add_edge(n.id, f"UPSAMPLE 0\n{self.deconv_layers[0].kernel_size}x{self.deconv_layers[0].in_channels}")
        return G
    
    def reset_activations(self, inputs=None):
        if inputs is None:
            inputs = self.get_constant_inputs([1,1])
        for n in self.nodes.values():
            n.value = torch.zeros(inputs.shape[1:]).to(self.device)
    
    def to(self, device):
        self.device = device
        for i, l in enumerate(self.conv_layers):
            self.conv_layers[i] = l.to(device)
        for i, l in enumerate(self.deconv_layers):
            self.deconv_layers[i] = l.to(device)
        for n in self.nodes.values():
            n.bias = n.bias.to(device)
            if n.value is not None:
                n.value = n.value.to(device)
        for c in self.connections.values():
            c.weight = torch.tensor(c.weight.item()).to(device)
            c.weight.requires_grad = True

    def __call__(self, X):
        return self.forward(X)
    
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
            labels={n:f"{n}\n{self.nodes[n].fn.__name__[:4]}\n{self.hidden_dim}xHxW"if n in self.nodes else n
                    for n in G.nodes(data=False) },
            node_size=800,
            font_size=6,
            node_shape='s',
            node_color=['lightsteelblue' if n in self.nodes else 'lightgreen' for n in G.nodes()  ]
            )
        plt.annotate('# params: ' + str(self.num_params), xy=(1.0, 1.0), xycoords='axes fraction', fontsize=12, ha='right', va='top')
        plt.show()
            
    def random_uniform(self, minval=-1, maxval=1):
        return np.random.uniform(minval, maxval)
    def random_normal(self, sigma=1.0):
        return np.random.normal(0, sigma)
    def random_activation(self,):
        idx = np.random.randint(0, len(all_activations))
        return all_activations[idx]
    
    def clone(self, new_id=True):
        """ Create a copy of this genome. """
        for p in self.params:
            p = p.detach()
        for _, n in self.nodes.items():
            n.value = None
        id = self.id if (not new_id) else type(self).get_id()
        child = copy.deepcopy(self)
        child.id = id
        return child
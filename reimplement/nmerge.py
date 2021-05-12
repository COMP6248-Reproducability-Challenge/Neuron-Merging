"""The function model_architecture was largely copied from the library torchsummary:

https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py"""

import torch
import torch.nn as nn
from collections import OrderedDict
import copy
from sklearn.metrics import pairwise_distances
import sys
sys.path.append('..')
from decompose import create_scaling_mat_ip_thres_bias
from decompose import create_scaling_mat_conv_thres_bn
import numpy as np

def model_architecture(model,
                       input_size,
                       batch_size=-1,
                       device=torch.device("cpu"),
                       dtypes=None):

    if dtypes == None:
        dtypes = [torch.FloatTensor for val in input_size]

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["object"] = module
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    """
    if isinstance(input_size, tuple):
        input_size = [input_size]
    """

    # batch_size of 2 for batchnorm
    """
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]
    """
    x = [torch.randn((1, *input_size))]

    # apply hooks
    summary = OrderedDict()
    hooks = []
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove hooks
    for h in hooks:
        h.remove()

    return summary

def get_output_channel_index(value, pruning_ratio=0, criterion='l1-norm'):

        output_channel_index = []

        if len(value.size()) :

            weight_vec = value.view(value.size()[0], -1)

            # l1-norm
            if criterion == 'l1-norm':
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().detach().numpy()
                arg_max = np.argsort(norm_np) 
                arg_max_rev = arg_max[::-1][:int((1-pruning_ratio)*len(arg_max))] # self.cfg[layer_id] is the number of filters kept for a given layer [300, 100] if pruning_ratio = 0.5 -> cfg = [150, 50] arg_max = [5,4,3]
                output_channel_index = sorted(arg_max_rev.tolist())
            
            # l2-norm
            elif criterion == 'l2-norm':
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().detach().numpy()
                arg_max = np.argsort(norm_np)
                arg_max_rev = arg_max[::-1][:int((1-pruning_ratio)*len(arg_max))]
                output_channel_index = sorted(arg_max_rev.tolist())

            # l2-GM
            """
            elif criterion == 'l2-GM':
                weight_vec = weight_vec.cpu().detach().numpy()
                matrix = distance.cdist(weight_vec, weight_vec, 'euclidean')
                similar_sum = np.sum(np.abs(matrix), axis=0)

                output_channel_index = np.argpartition(similar_sum, -self.cfg[layer_id])[-self.cfg[layer_id]:]
            """

        return output_channel_index
    
def pruning_mask(weight, threshold = 0.9):
    cosine_dist = pairwise_distances(weight.flatten(1), metric="cosine")
    sim = torch.from_numpy(1-cosine_dist).abs() > threshold
    triangle = torch.arange(0,1,1/sim.shape[1]).repeat(sim.shape[0],1)
    triangle = triangle - triangle.t()
    triangle = triangle > 0
    mask = ~(sim & triangle).any(1)
    return mask

def merge_on_layers(layers,
                    threshold=0.9,
                    epochs = 100,
                    max_acc = 5e-3):
    old_model = nn.Sequential(*(item["object"] for item in layers))
    new_model = copy.deepcopy(old_model)
    
    #apply mask to first and last layer
    mask = pruning_mask(new_model[0].weight.data,threshold=threshold)
    new_model[0].weight.data = new_model[0].weight.data[mask]
    new_model[-1].weight.data = new_model[-1].weight.data[:,mask]
    if hasattr(new_model[0], "bias"):
        new_model[0].bias.data = new_model[0].bias.data[mask]
    
    
    #change in and out sizes
    if isinstance(new_model[0], nn.Linear):
        new_model[0].out_features -= (~mask).sum()
        new_model[-1].in_features -= (~mask).sum()
        deg_freedom = 1 + min(int(new_model[0].out_features),
                              int(new_model[0].in_features))
    elif isinstance(new_model[0], nn.Conv2d):
        new_model[0].out_channels -= (~mask).sum()
        new_model[-1].in_channels -= (~mask).sum()
        deg_freedom = 1 + min(int(new_model[0].out_channels),
                              int(new_model[0].in_channels))
    
    
    #train
    samples = torch.randn((deg_freedom, *layers[0]["input_shape"][1:]))
    features = torch.cat((samples, -samples))
    targets = old_model(features).detach()
    
    loss_func = torch.nn.MSELoss()
    magnitude = loss_func(targets, torch.zeros(targets.shape))
    optimiser = torch.optim.SGD(new_model.parameters(), lr=1e-2, momentum=0.9)

    for _ in range(epochs):
        optimiser.zero_grad()
        loss = loss_func(new_model(features), targets)/magnitude
        if loss < max_acc:
            break
        loss.backward()
        optimiser.step()

    return [item for item in new_model]
    
def assign_weights(old_obj, new_obj):
    old_obj.weight.data = new_obj.weight.data
    if hasattr(old_obj, "bias"):
        old_obj.bias.data = new_obj.bias.data
        
    if isinstance(old_obj, nn.Linear):
        old_obj.out_features = new_obj.out_features
        old_obj.in_features = new_obj.in_features
    elif isinstance(old_obj, nn.Conv2d):
        old_obj.out_channels = new_obj.out_channels
        old_obj.in_channels = new_obj.in_channels
    
def neuron_merge(model,
                 input_size,
                 pruning_ratio = 0.5,
                 threshold=0.9,
                 epochs = 100,
                 max_acc = 5e-3,
                 pruning_criterion='l1-norm',
                 model_type = "merge", #original, prune, or merge
                 dif_method = False):
    new_model = copy.deepcopy(model)
    arch = list(model_architecture(new_model, input_size).items())[:-1]
    trainable = [ind for ind, val in enumerate(arch) 
                 if isinstance(val[1]["object"],  (nn.Linear, nn.Conv2d))]
                 
    supported = (nn.Linear,
                 nn.Conv2d,
                 nn.MaxPool2d,
                 nn.Dropout,
                 nn.ReLU,
                 nn.Identity)
    check_type = lambda val: isinstance(val["object"], supported)
    check_dropout = lambda val: not isinstance(val["object"], nn.Dropout)

    for start, end in zip(trainable[:-1], trainable[1:]):
        #get layers, and check if supported
        layers = list(filter(check_dropout, list(zip(*arch))[1][start:end+1]))
        if not all(map(check_type, layers)):
            continue
        
        #find merged values
        if dif_method:
            merged_layers = merge_on_layers(layers,
                                            threshold=threshold,
                                            epochs = epochs,
                                            max_acc = max_acc)
            #assign values to model
            assign_weights(layers[0]["object"], merged_layers[0])
            assign_weights(layers[-1]["object"], merged_layers[-1])
        else:
             if isinstance(layers[0]["object"], nn.Linear): 
                    

                        
                    weight = layers[0]["object"].weight.cpu().detach().numpy()
                    bias = layers[0]["object"].bias.cpu().detach().numpy()


                    bias_reshaped = bias.reshape(bias.shape[0],-1)
                    concat_weight = np.concatenate([weight, bias_reshaped], axis = 1)
                    
                    output_channel_index = get_output_channel_index(torch.from_numpy(concat_weight), pruning_ratio=pruning_ratio, criterion=pruning_criterion)

                    # make scale matrix with bias
                    x = create_scaling_mat_ip_thres_bias(concat_weight, np.array(output_channel_index), threshold, model_type) # merge, prune
                    z = torch.from_numpy(x).type(dtype=torch.float)

                    # pruned
                    layers[0]["object"].weight.data = layers[0]["object"].weight[output_channel_index,:]
                    layers[0]["object"].in_features = layers[0]["object"].weight.data.shape[0]
                    layers[0]["object"].out_features = layers[0]["object"].weight.data.shape[-1]
                    
                    # update next input channel
                    input_channel_index = output_channel_index                    
                    layers[-1]["object"].weight.data = layers[-1]["object"].weight @ z
                    layers[-1]["object"].in_features = layers[-1]["object"].weight.data.shape[0]
                    layers[-1]["object"].out_features = layers[-1]["object"].weight.data.shape[-1]

    return new_model
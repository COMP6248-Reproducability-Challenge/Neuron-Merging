"""The function model_architecture was largely copied from the library torchsummary:

https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py"""

import torch
import torch.nn as nn
from collections import OrderedDict
import copy
from sklearn.metrics import pairwise_distances

def model_architecture(model,
                       input_size,
                       batch_size=-1,
                       device=torch.device("cpu"),
                       dtypes=None):

    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

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
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

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
                 threshold=0.9,
                 epochs = 100,
                 max_acc = 5e-3):
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
    i=0
    for start, end in zip(trainable[:-1], trainable[1:]):
        i+=1
        print(i)
        print(start)
        print(end)
        print(new_model)
        #get layers, and check if supported
        layers = list(filter(check_dropout, list(zip(*arch))[1][start:end+1]))
        if not all(map(check_type, layers)):
            continue
        
        #find merged values
        merged_layers = merge_on_layers(layers,
                                        threshold=threshold,
                                        epochs = epochs,
                                        max_acc = max_acc)
        #assign values to model
        assign_weights(layers[0]["object"], merged_layers[0])
        assign_weights(layers[-1]["object"], merged_layers[-1])
    
    print("end")
    return new_model
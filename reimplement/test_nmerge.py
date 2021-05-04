import torch
import torchvision
from nmerge import *
import pprint


def test():
    model = torchvision.models.alexnet(pretrained=True)  

    print(model)  
    new_model = neuron_merge(model, (3, 224, 224), threshold=0.1)
    print(new_model)
    
    
if __name__ == "__main__":
    test()
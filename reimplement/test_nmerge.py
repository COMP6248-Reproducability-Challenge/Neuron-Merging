import torch
import torchvision
from nmerge import *
import pprint
import sys
sys.path.append('..')
import models

def test():
    #model = torchvision.models.alexnet(pretrained=True)
    #input_shape = (3, 224, 224)    
    #model = torchvision.models.lenet(pretrained=True)

    cfg = [300,100]    
    model = models.LeNet_300_100(bias_flag=True, cfg=cfg)
    input_shape = [784]
    
    new_model = neuron_merge(model, input_shape, threshold=0.5)
    print(model)
    print(new_model)
    
    
if __name__ == "__main__":
    test()
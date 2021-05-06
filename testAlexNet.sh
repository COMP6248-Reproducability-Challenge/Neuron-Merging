 python main.py --arch Alexnet-CIFAR --dataset cifar100 --retrain --target ip --criterion l1-norm --model-type merge --pretrained saved_models/alexnet-cifar-sd.pth  --pruning-ratio 0
 .5 --evaluate --no-cuda

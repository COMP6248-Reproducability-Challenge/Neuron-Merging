#!/bin/bash



AlexNet_original(){
echo -e "\nAlexNet original\n"
python main.py \
   --arch AlexNet_CIFAR100 \
   --dataset cifar100 \
   --pretrained saved_models/alexnet-cifar.pth \
   --implementation original \
   --evaluate
}

AlexNet_prune(){
echo -e "\nAlexNet prune $1 $2\n"
python main.py \
   --arch AlexNet_CIFAR100 \
   --dataset cifar100 \
   --retrain \
   --target ip \
   --criterion $1 \
   --model-type prune \
   --pretrained saved_models/alexnet-cifar.pth \
   --pruning-ratio $2 \
   --implementation original \
   --evaluate
}

AlexNet_merge(){
echo -e "\nAlexNet merge $1 $2 $3\n"
python main.py \
   --arch AlexNet_CIFAR100 \
   --dataset cifar100 \
   --retrain \
   --target ip \
   --criterion $1 \
   --model-type merge \
   --pretrained saved_models/alexnet-cifar.pth \
   --threshold 0.1 \
   --pruning-ratio $2 \
   --lamda 0.7 \
   --implementation $3 \
   --evaluate
}


help() {
    echo "AlexNet_CIFAR100.sh [OPTIONS]"
    echo "    -h		help."
    echo "    -t ARG    model type: original | prune | merge (default: original)."
    echo "    -c ARG    criterion : l1-norm | l2-norm | l2-GM (default: l1-norm)."
    echo "    -r ARG 		pruning ratio : (default: 0.5)."
    echo "    -i ARG    implementation: original | reimplementation (default: original)"
    exit 0
}

model_type=original
criterion=l1-norm
pruning_ratio=0.5
implementation=original

while getopts "t:c:r:i:h" opt
do
    case $opt in
	      t) model_type=$OPTARG
          ;;
        c) criterion=$OPTARG
          ;;
        r) pruning_ratio=$OPTARG
          ;;
        i) implementation=$OPTARG
          ;;
        h) help ;;
        ?) help ;;
    esac
done


AlexNet_$model_type $criterion $pruning_ratio $implementation
#!/bin/bash



VGG16_original(){
echo -e "\nVGG16 original\n"
python main.py \
   --arch VGG \
   --dataset cifar100 \
   --pretrained saved_models/VGG.cifar100.original.pth.tar \
   --implementation original \
   --evaluate
}

VGG16_prune(){
echo -e "\nVGG16 prune $1\n"
python main.py \
   --arch VGG \
   --dataset cifar100 \
   --retrain \
   --target conv \
   --criterion $1 \
   --model-type prune \
   --pretrained saved_models/VGG.cifar100.original.pth.tar \
   --implementation original \
   --evaluate
}

VGG16_merge(){
echo -e "\nVGG16 merge $1 $2\n"
python main.py \
   --arch VGG \
   --dataset cifar100 \
   --retrain \
   --target conv \
   --criterion $1 \
   --model-type merge \
   --pretrained saved_models/VGG.cifar100.original.pth.tar \
   --threshold 0.1 \
   --lamda 0.7 \
   --implementation $2 \
   --evaluate
}


help() {
    echo "VGG16_CIFAR100.sh [OPTIONS]"
    echo "    -h		help."
    echo "    -t ARG    model type: original | prune | merge (default: original)."
    echo "    -c ARG    criterion : l1-norm | l2-norm | l2-GM (default: l1-norm)."
    echo "    -i ARG    implementation: original | reimplementation (default: original)"
    exit 0
}

model_type=original
criterion=l1-norm
implementation=original

while getopts "t:c:i:h" opt
do
    case $opt in
	t) model_type=$OPTARG
          ;;
        c) criterion=$OPTARG
          ;;
        i) implementation=$OPTARG
          ;;
        h) help ;;
        ?) help ;;
    esac
done


VGG16_$model_type $criterion $implementation
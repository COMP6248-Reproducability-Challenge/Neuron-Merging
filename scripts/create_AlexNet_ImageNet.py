import wget

url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
wget.download(url, 'saved_models/alexnet-imagenet.pth')
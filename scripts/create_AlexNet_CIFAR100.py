import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class Alexnet(nn.Module):

    def __init__(self, num_classes, cfg):
        if cfg == None:
            cfg = [4096, 4096]
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, cfg[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(cfg[0], cfg[1]),
            nn.ReLU(inplace=True),
            nn.Linear(cfg[1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False, num_workers=2)

model = Alexnet(num_classes=100,cfg=None)
model.train()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

epochs=320

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Learning rate: ',optimizer.param_groups[0]['lr'])
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
    scheduler.step()

# model = torch.load('alexnet-cifar.pth',map_location=torch.device('cpu'))
model.eval()

criterion = nn.CrossEntropyLoss()

test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    output = model(data)
    test_loss += criterion(output, target).data
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

acc = 100. * float(correct) / len(test_loader.dataset)

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    test_loss * 512, correct, len(test_loader.dataset),
    100. * float(correct) / len(test_loader.dataset)))

print('==> Saving model ...')
state = {
        'acc': acc,
        'state_dict': model.state_dict(),
        }
for key in state['state_dict'].keys():
    if 'module' in key:
        print(key)
        state['state_dict'][key.replace('module.', '')] = \
                state['state_dict'].pop(key)

torch.save(state,'saved_models/alexnet-cifar.pth')
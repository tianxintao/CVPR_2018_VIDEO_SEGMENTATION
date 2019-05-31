import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from dataloader import VideoSegmentationDataset
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
#torch.cuda.empty_cache() 
# Hyper parameters
num_epochs = 5
num_classes = 35
batch_size = 1
learning_rate = 0.003

## MNIST dataset
sampledTrainFolderPath = '../train_color_sample/'
sampledLabelFolderPath = '../train_label_sample/'
train_dataset = VideoSegmentationDataset(sampledTrainFolderPath,
                                sampledLabelFolderPath)

#test_dataset = torchvision.datasets.MNIST(root='../data/',
#                                          train=False, 
#                                          transform=transforms.ToTensor())
#
## Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size)

#test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                          batch_size=batch_size, 
#                                          shuffle=False)

#im1 = train_dataset.data[0]
#
#print(train_dataset.data[0].shape)
# Convolutional neural network (two convolutional layers)
class FCN(nn.Module):
    def __init__(self, num_classes=35):
        super(FCN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(36, 64, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU())
        
        self.layer7 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

#        self.fc = nn.Linear(7*7*32, num_classes)
        self.score_fr = nn.Sequential(
                nn.Conv2d(256,35,kernel_size=1,padding=0))
        
        self.upsampling = nn.Sequential(
                nn.ConvTranspose2d(35,35,kernel_size=72,stride=28))
        
    def forward(self, x):
        out = self.layer1(x.view(-1,3,2710,3384))
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
#        out = out.reshape(out.size(0), -1)
#        out = self.fc(out)
        out = self.score_fr(out)
        out = self.upsampling(out)
        return out

model = FCN().to(device)
print(summary(model,(3,2710,3384)))


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
#         print(data)
        images = data['image'].type('torch.DoubleTensor').to(device)
        labels = data['label'].type('torch.DoubleTensor').to(device)
        print(1)
        # Forward pass
        outputs = model(images.double())
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

## Test the model
#model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
#with torch.no_grad():
#    correct = 0
#    total = 0
#    for images, labels in test_loader:
#        images = images.to(device)
#        labels = labels.to(device)
#        outputs = model(images)
#        _, predicted = torch.max(outputs.data, 1)
#        total += labels.size(0)
#        correct += (predicted == labels).sum().item()
#
#    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
#
## Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')# -*- coding: utf-8 -*-


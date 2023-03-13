import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#prepare data
batch_size=64
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))
])
train_dataset=datasets.MNIST(root='../dataset/mnist',train=True,transform=transform,download=True)
train_loader=DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
test_data=datasets.MNIST(root='../dataset/mnist',train=False,transform=transform,download=True)
test_loader=DataLoader(dataset=test_data,shuffle=False,batch_size=batch_size)

class ResidualBlock(torch.nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.channels=channels
        self.conv1=torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)#保证bcwh都不变=kernel配padding
        self.conv2=torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x):#流通过这个块
        y=F.relu(self.conv1(x))
        y=self.conv2(y)
        return F.relu(x+y)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(1,16,kernel_size=5)
        self.conv2=torch.nn.Conv2d(16,32,kernel_size=5)
        self.mp=torch.nn.MaxPool2d(2)

        self.rblock1=ResidualBlock(16)
        self.rblock2=ResidualBlock(32)

        self.fc=torch.nn.Linear(512,10)

    def forward(self,x):
        in_size=x.size(0)
        x=self.mp(F.relu(self.conv1(x)))
        x=self.rblock1(x)
        x=self.mp(F.relu(self.conv2(x)))
        x=self.rblock2(x)
        x=x.view(in_size,-1)
        x=self.fc(x)
        return x


model=Net()

#loss and optim
criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)


#train
def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        output=model(inputs)
        optimizer.zero_grad()
        loss=criterion(output,target)

        running_loss+=loss.item()

        loss.backward()

        optimizer.step()

        if(batch_idx%300==299):
            print('[%d %5d]   loss:%.3f' %(epoch+1,batch_idx+1,running_loss/300))
            running_loss=0.0
def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,label=data

            total+=label.size(0)

            output=model(images)
            _,pred=torch.max(output.data,dim=1)

            correct+=(pred==label).sum().item()

            acc=correct/total

    print('Accraccy on test :%.3f'%(acc*100))
    return acc

if __name__ =='__main__':
    epoch_list=[]
    acc_list=[]
    for epoch in range(2):
        train(epoch)
        acc=test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list,acc_list)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()
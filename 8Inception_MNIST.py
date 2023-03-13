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

#InceptionA model
#一个块是一个class   块可以嵌套
class InceptionA(torch.nn.Module):
    def __init__(self,in_channel):
        super(InceptionA,self).__init__()
        #定义第i个分支里有什么'层'
        self.branch1_2=torch.nn.Conv2d(in_channel,24,kernel_size=1)

        self.branch2=torch.nn.Conv2d(in_channel,16,kernel_size=1)

        self.branch3_1=torch.nn.Conv2d(in_channel,16,kernel_size=1)
        self.branch3_2=torch.nn.Conv2d(16,24,kernel_size=5,padding=2)

        self.branch4_1=torch.nn.Conv2d(in_channel,16,kernel_size=1)
        self.branch4_2=torch.nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch4_3=torch.nn.Conv2d(24,24,kernel_size=3,padding=1)

    def forward(self,x):
        #每次流通 第i个分支里的所有层
        branch1=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch1=self.branch1_2(branch1)

        branch2=self.branch2(x)

        branch3=self.branch3_1(x)
        branch3=self.branch3_2(branch3)

        branch4=self.branch4_1(x)
        branch4=self.branch4_2(branch4)
        branch4=self.branch4_3(branch4)

        output=[branch1,branch2,branch3,branch4]
        return torch.cat(output,dim=1)

#advanced CNN model which contains InceptionA model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(88,20,kernel_size=5)

        self.incep1=InceptionA(in_channel=10)
        self.incep2=InceptionA(in_channel=20)

        self.max_pooling=torch.nn.MaxPool2d(2)
        self.fc=torch.nn.Linear(1408,10)

    def forward(self,x):
        in_size=x.size(0)#batch_size
        x=F.relu(self.max_pooling(self.conv1(x)))
        x=self.incep1(x)
        x=F.relu(self.max_pooling(self.conv2(x)))
        x=self.incep2(x)
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






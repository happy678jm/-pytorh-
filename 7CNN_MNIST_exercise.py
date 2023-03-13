import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#prepare data
batch_size=64
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))
])

train_dataset=datasets.MNIST(root='../dataset/mnist',train=True,download=True,transform=transform)
train_loader=DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
test_dataset=datasets.MNIST(root='../dataset/mnist',train=False,download=True,transform=transform)
test_loader=DataLoader(dataset=test_dataset,shuffle=False,batch_size=batch_size)

#model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(10,10,kernel_size=5)
        self.conv3=torch.nn.Conv2d(20,30,kernel_size=3)
        self.linear1=torch.nn.Linear(30,20)
        self.linear2=torch.nn.Linear(20,15)
        self.linear3=torch.nn.Linear(15,10)
        self.pooling=torch.nn.MaxPool2d(2)
    def forward(self,x):
        batch_size=x.size(0)
        x=self.pooling(F.relu(self.conv1(x)))
        x=self.pooling(F.relu(self.conv2(x)))
        x=self.pooling(F.relu(self.conv3(x)))
        x=x.view(batch_size,-1)
        x=self.linear1(x)
        x=self.linear2(x)
        x=self.linear3(x)
        return x
model=Net()
#loss and optimzer
criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

#train
def train(epoch):
    ranning_loss=0.0
    for batch_index,data in enumerate(train_loader,1):
        inputs,target=data
        outputs=model(inputs)
        optimizer.zero_grad()
        loss=criterion(outputs,target)

        loss.backward()
        optimizer.step()

        ranning_loss+=loss.item()

        if batch_index %300==299:
            print('[%d %5d] loss:%.3f' %(epoch+1,batch_index+1,ranning_loss/300))
            ranning_loss=0.0

def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            image,label=data
            output=model(image)
            _,pred=torch.max(output.data,dim=1)
            total+=label.size(0)
            correct+=(pred==label).sum().item()
            acc=correct/total
    print('Accurracy on test set:%d %%'%(100*correct/total))
    return acc

if __name__=='__main__':
    acc_list=[]
    epoch_list=[]
    for epoch in range(3):
        train(epoch)
        acc=test()
        acc_list.append(acc)
        epoch_list.append(epoch)

    plt.plot(epoch_list,acc_list)
    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.show()


import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision import datasets
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
train_dataset=datasets.MNIST(root='../dataset/mnist/',train=True,download=True,transform=transform)
train_loader=DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
test_dataset=datasets.MNIST(root='../dataset/mnist/',train=False,download=True,transform=transform)
test_loader=DataLoader(dataset=test_dataset,shuffle=False,batch_size=batch_size)

#model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling=torch.nn.MaxPool2d(2)
        self.linear=torch.nn.Linear(320,10)

    def forward(self,x):
        batch_size=x.size(0)
        x=F.relu(self.pooling(self.conv1(x)))
        x=F.relu(self.pooling(self.conv2(x)))
        x=x.view(batch_size,-1)#变成向量 竖条 之后 再进入线性
        x=self.linear(x)
        return x
model=Net()

#loss and optimizer
criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

#train
def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        input,target=data

        optimizer.zero_grad()

        output=model(input)
        loss=criterion(output,target)

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss:%.3f' %(epoch+1,batch_idx+1,running_loss/300))
            running_loss=0.0


def test():
    #tmd不同for了好吧 加了个for之后出错好久
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            image,label=data
            output=model(image)
            _,predict=torch.max(output.data,dim=1)
            total+=label.size(0)
            correct+=(predict==label).sum().item()
            acc=correct/total
    print("Accuarcy on test set: %d %%" %(100*correct/total))
    return acc


if __name__=='__main__':
    acc_list = []
    for epoch in range(10):
        train(epoch)
        acc=test()
        acc_list.append(acc)
    plt.plot(acc_list)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()

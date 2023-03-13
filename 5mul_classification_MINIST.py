import torch
from torchvision import transforms#对图像进行原始的数据处理的工具
from torchvision import datasets#获取数据
from torch.utils.data import DataLoader#加载数据
import torch.nn.functional as F#与全联接的relu激活函数有关
import torch.optim as optim#与优化器有关
#prepare data
batch_size=64#GPU对2的幂可发挥更加性能，16。32。64。128。。。。更好
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))
])
train_dataset=datasets.MNIST(root='../dataset/mnist/',train=True,download=True,transform=transform)
train_loader=DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
test_dataset=datasets.MNIST(root='../dataset/mnist',train=False,download=True,transform=transform)
test_loader=DataLoader(dataset=test_dataset,shuffle=False,batch_size=batch_size)
#model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1=torch.nn.Linear(784,512)#一张图片是28x28的像素，28x28=784个像素点（数值），把一张图片按行-拼接成一大行----，共784个值，即784列
        self.l2=torch.nn.Linear(512,256)
        self.l3=torch.nn.Linear(256,128)
        self.l4=torch.nn.Linear(128,64)
        self.l5=torch.nn.Linear(64,10)


    def forward(self,x):
        x=x.view(-1,784)#reshape  -1是自动获取mini——batch
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=F.relu(self.l4(x))
        x=self.l5(x)
        return x
model=Net()

criteriom=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader ,0):
        inputs,target=data
        optimizer.zero_grad()

        outputs=model(inputs)
        loss=criteriom(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idx%300==299:#每300个minibacth打印一次
            print('[%d,%5d] loss:%.3f'%(epoch+1,batch_idx+1,running_loss/300))
            running_loss=0.0

def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            _,predict=torch.max(outputs.data,dim=1)
            total+=labels.size(0)
            correct+=(predict==labels).sum().item()
    print('Accuracy on test set:%d %%' %(100*correct/total))







if __name__ =='__main__':
    for epoch in range(2):
        train(epoch)
        test()


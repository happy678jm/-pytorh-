import numpy as np
import torch
import matplotlib.pyplot as plt
#prepare data 手动的，没有涉及mini-batch，而是全部batch
xy=np.loadtxt('diabetes.csv',delimiter=',',dtype=np.float32)
x_data=torch.from_numpy(xy[:,:-1])
y_data=torch.from_numpy(xy[:,[-1]])

#define model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()
        self.relu=torch.nn.ReLU()

    def forward(self,x):
        x=self.relu(self.linear1(x))
        x=self.relu(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x
model=Model()

#loss and optimizer
criterion=torch.nn.BCELoss(size_average=True)#因为2分类任务，所以要用BCEloss
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)


epoch_list=[]
loss_list=[]

#trainning cycle
for epoch in range(200):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())

    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

#test
x_test=torch.Tensor([1,85,66,29,0,26.6,0.351,31])
y_pred=model(x_test)
print(y_pred.data)

plt.plot(epoch_list,loss_list)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()



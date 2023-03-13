import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

#prepare data
x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[0],[0],[1]])

#design model using class
class LogisticRegressionModel(torch.nn.Module):
#1 init
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)
#2 forward
    def forward(self,x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred
model=LogisticRegressionModel()

#construct loss and optimizer
criterion=torch.nn.BCELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#training cycle
for epoch in range(1000):
    #forward
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())
    #backward
    optimizer.zero_grad()
    loss.backward()
    #update
    optimizer.step()

x=np.linspace(0,10,200)
x_t=torch.Tensor(x).view((200,1))
y_t=model(x_t)
y=y_t.data.numpy()
plt.plot(x,y)
plt.xlabel('Hours')
plt.ylabel("P(Pass)")
plt.grid()
plt.show()

#test
x_test=torch.Tensor([4.0])
y_test=model(x_test)
print(y_test)
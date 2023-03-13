import torch
import matplotlib.pyplot as plt
#prepare dataset
x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[2.0],[4.0],[6.0]])

#design model using class
class LinearModel(torch.nn.Module):
    #1 init
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)
    #2 forward
    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred
#model
model=LinearModel()

#loss and optimizer
criterion=torch.nn.MSELoss(size_average=False)
#optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
#optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
#optimizer=torch.optim.Adamax(model.parameters(),lr=0.01)
#optimizer=torch.optim.ASGD(model.parameters(),lr=0.01)
#optimizer=torch.optim.LBFGS(model.parameters(),lr=0.01)   用法不一样
#optimizer=torch.optim.RMSprop(model.parameters(),lr=0.01)
#optimizer=torch.optim.Rprop(model.parameters(),lr=0.01)
optimizer=torch.optim.Adagrad(model.parameters(),lr=0.01)

loss_list=[]
epoch_list=[]
#training cycle =forward+backward+update
for epoch in range(1000):
    #yhat loss print 0 backward step
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())

    loss_list.append(loss.item())
    epoch_list.append(epoch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'W',{model.linear.weight.item()})
print(f'b',{model.linear.bias.item()})

plt.plot(epoch_list,loss_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.title('Adagrad')
plt.show()

#test
x_test=torch.Tensor([4.0])
y_test=model(x_test)
print(y_test.data)


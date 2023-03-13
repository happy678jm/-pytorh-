import torch
input_size=4
hidden_size=4
batch_size=1

#1 string to vector
idx2char=['e','h','l','o']
x_data=[1,0,2,2,3]#hello
y_data=[3,1,2,3,2]#ohlol
one_hot_lookup=[
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
]
x_one_hot=[one_hot_lookup[x] for x in x_data]
inputs=torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
labels=torch.LongTensor(y_data).view(-1,1)
#print(labels)  需要用cell的for循环一个一个label比对 和RNN不同

class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super(Model,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.rnncell=torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)

    def forward(self,input,hidden):
        hidden=self.rnncell(input,hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)
net=Model(input_size,hidden_size,batch_size)

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.1)

for epoch in range(15):
    loss=0
    optimizer.zero_grad()
    hidden=net.init_hidden()
    print('Predicted string:',end='')
    for input,label in zip(inputs,labels):
        hidden=net(input,hidden)
        loss+=criterion(hidden,label)
        _,idx=hidden.max(dim=1)#batch_sizexhidden_size
        print(idx2char[idx.item()],end='')
    loss.backward()
    optimizer.step()
    print('  ,Epoch:',epoch+1,'loss: %.4f:'%(loss.item()))




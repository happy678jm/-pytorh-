#直接调用GRU
import torch

input_size=4
hidden_size=4
num_layers=1
batch_size=1
seq_len=5

idx2char=['e','h','l','o']
x_data=[1,0,2,2,3]#hello
y_data=[3,1,2,3,2]#ohlol

one_hot_lookup=[
    [1,0,0,0],#e
    [0,1,0,0],#h
    [0,0,1,0],#l
    [0,0,0,1]#o
]


x_one_hot=[one_hot_lookup[x] for x in x_data]
inputs=torch.Tensor(x_one_hot).view(seq_len,batch_size,input_size)
labels=torch.LongTensor(y_data)
#他不是一个输入进cell输出一个hidden，而是输出一批hidden，不同分割成一个个label，而是一个向量的label
#rnncell 是用for循环zip一对一对来算损失，损失再相加，而这里是组装成一个向量
#print(labels)
class model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,num_layers=1):
        super(model,self).__init__()
        self.num_layers=num_layers
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        self.gru=torch.nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bias=False)

    def forward(self, inputs):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, h = self.gru(inputs,hidden)
        return out.view(-1, hidden_size)
    # 拉成一个与标签label对应的向量

net=model(input_size,hidden_size,batch_size,num_layers)

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.5)

for epoch in range(25):
    optimizer.zero_grad()
    outputs=net(inputs)
    loss=criterion(outputs,labels)
    loss.backward()
    optimizer.step()

    _,idx=outputs.max(dim=1) #value idx
    idx=idx.data.numpy()
    print('Predit:',''.join([idx2char[x] for x in idx]),end='')
    print('  Epoch:',epoch+1,'Loss: %.3f' %(loss.item()))

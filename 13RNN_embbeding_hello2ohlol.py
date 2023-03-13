import torch
num_class=4
input_size=4
hidden_size=8
embedding_size=10
num_layers=2
batch_size=1
seq_len=5

idx2char=['e','h','l','o']
x_data=[1,0,2,2,3]#batchxseq_len
y_data=[3,1,2,3,2]
inputs=torch.LongTensor(x_data).view(-1,seq_len)#batchsize,seqlen
#print(inputs.shape)
labels=torch.LongTensor(y_data)

class model(torch.nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.emb=torch.nn.Embedding(input_size,embedding_size)
        self.rnn=torch.nn.RNN(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True)
        self.fc=torch.nn.Linear(hidden_size,num_class)

    def forward(self,x):
        hidden=torch.zeros(num_layers,x.size(0),hidden_size)
        x=self.emb(x)
        x,_=self.rnn(x,hidden)
        x=self.fc(x)
        return x.view(-1,num_class)

net=model()

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    output=net(inputs)
    loss=criterion(output,labels)
    loss.backward()
    optimizer.step()

    _,idx=output.max(dim=1)
    idx=idx.data.numpy()
    print('Predicted:',''.join([idx2char[x] for x in idx]),end='')
    print('. Epoch: %d Loss:%.3f'%(epoch+1,loss.item()))

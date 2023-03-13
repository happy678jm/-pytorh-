import torch
batch_size=1
seq_len=3
input_size=4
hidden_size=2
num_layers=1

#1.RNN cell 自己对seq个写循环
cell=torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)

dataset=torch.randn(seq_len,batch_size,input_size)#(seq,batch,feature)  输入整体数据集
hidden=torch.zeros(batch_size,hidden_size)#输入 ho
for idx,input in enumerate(dataset):#按dataset第一w维拿 x1\x2\x3...xseq

    print('='*20,idx,'='*20)
    print('input size',input.shape)

    hidden=cell(input,hidden)#cell自动算

    print('output size',hidden.shape)
    print(hidden)
#RNN 直接调用一步到位算出
#cell=torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size,num_layer=num_layers)
cell_setting=torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)

#inputs=torch.randn(seq_len,batch_size,input_size)
input=torch.randn(batch_size,seq_len,input_size)
hidden_input=torch.zeros(num_layers,batch_size,hidden_size)#多了个num——layers

out,hidden =cell(inputs,hidden_input)

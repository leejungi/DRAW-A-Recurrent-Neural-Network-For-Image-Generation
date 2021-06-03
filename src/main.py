import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor 
import matplotlib.pyplot as plt

from datasets import Dataset
from model import DRAW

Train = True
Test = True

device = 'cuda'

epochs = 20
learning_rate = 1e-3
batch_size = 64
weight_decay = 0

T = 10 # time size
A = 28 # height
B = 28 # width
rep_dim = 10 # representation dimension
N = 5 # patch size
dec_size = 256
enc_size = 256

def train(train_loader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay,betas=(0.5,0.999))
    
    # Train
    total_batch = len(train_loader)
    batch_num = 0
    for epoch in range(epochs):
        avg_loss=0
        model.train()
        for X, _ in train_loader:
            X = X.to(device)
            
            optimizer.zero_grad()
            
            loss = model.loss(X)
            loss.backward()#Gradient calculation
            optimizer.step()#Gradient update
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            avg_loss += loss/(total_batch*batch_size)
            batch_num +=1

        avg_loss = float(avg_loss.to('cpu').detach())
        print("Epoch: {}/{} Avg loss: {}".format(epoch, epochs, avg_loss))

def generate(model):
    model.eval()
    with torch.no_grad():
        img=model.generate(64)
        t_img = img[-1] #Last Time
        for i in range(T):
            t_img = img[i]
            t_img = t_img.reshape((8,8,28,28))
            t_img = t_img.transpose((0,2,1,3))
            t_img = t_img.reshape((8*28, 8*28))
            plt.matshow(t_img, cmap=plt.cm.gray)
            plt.savefig('./image/T_{}.png'.format(i))

def main():
    
    train_data = MNIST(root='./', train=True, download=True, transform=ToTensor())
    # test_data = MNIST(root='./', train=False, download=True, transform=ToTensor())
         
    train_dset = Dataset(train_data)
    # test_dset = Dataset(test_data)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=False)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dset,
    #                                           batch_size=batch_size,
    #                                           shuffle=False,
    #                                           drop_last=False)
    
    model = DRAW(T,A,B,rep_dim,N,dec_size,enc_size,device).to(device)


    if Train == True:
        train(train_loader, model)
        generate(model)

if __name__ =="__main__":
    main()

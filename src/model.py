import torch.nn as nn
import torch
import numpy as np

class DRAW(nn.Module):
    def __init__(self,T,A,B,rep_dim,N,dec_size,enc_size, device,atten=True):
        super(DRAW,self).__init__()
        self.T = T
        # self.batch_size = batch_size
        self.A = A
        self.B = B
        self.rep_dim = rep_dim
        self.N = N
        self.dec_size = dec_size
        self.enc_size = enc_size

        
        self.logsigma = [0] * T
        self.sigma = [0] * T
        self.mu = [0] * T
        self.device= device
        self.atten=True
        self.eps = 1e-9
        
        self.encoder = nn.LSTMCell(2 * N * N + dec_size, enc_size)
        
        self.mu_linear = nn.Linear(dec_size, rep_dim)
        self.sigma_linear = nn.Linear(dec_size, rep_dim)

        self.decoder = nn.LSTMCell(rep_dim,dec_size)
        
        self.read_linear = nn.Linear(dec_size,5)
        if self.atten == True:
            self.write_linear = nn.Linear(dec_size,N*N)
        else:
            self.write_linear = nn.Linear(dec_size,self.A*self.B)

        self.sigmoid = nn.Sigmoid()
        
        self.criterion = nn.BCELoss()
        
    def forward(self, x):
        self.batch_size = x.size()[0]
        
        h_enc_out = torch.zeros(self.batch_size, self.enc_size).to(self.device)
        h_dec_out = torch.zeros(self.batch_size,self.dec_size).to(self.device)
        
        enc_hidden = torch.zeros(self.batch_size,self.enc_size).to(self.device)
        dec_hidden = torch.zeros(self.batch_size, self.dec_size).to(self.device)
        
        c = torch.zeros(self.T, self.batch_size,self.A * self.B).to(self.device)
        
        for t in range(self.T):
            pre_c = c[0] if t==0 else c[t-1]
            x_hat = x - self.sigmoid(pre_c)
            
            r_t = self.read(x, x_hat, h_dec_out)
            h_enc_out, enc_hidden = self.encoder(torch.cat((r_t, h_dec_out),1), (h_enc_out, enc_hidden))
            z_t, self.mu[t], self.logsigma[t], self.sigma[t] = self.Q(h_enc_out)
            h_dec_out, dec_hidden = self.decoder(z_t, (h_dec_out, dec_hidden))
            c[t] = pre_c + self.write(h_dec_out)
            
        return c
    
    def read(self, x, x_hat, h_dec):
        if self.atten==True:
            Fx, Fy, gamma = self.get_filter(h_dec)
            Fx = Fx.transpose(2,1)
            
            x = x.view(-1,self.B,self.A)
            x = Fy.bmm(x.bmm(Fx)).view(-1,self.N*self.N)
            x = x * gamma.view(-1,1).expand_as(x)
            
            x_hat = x_hat.view(-1,self.B,self.A)
            x_hat = Fy.bmm(x_hat.bmm(Fx)).view(-1,self.N*self.N)
            x_hat = x_hat * gamma.view(-1,1).expand_as(x_hat)
            
            return torch.cat((x,x_hat),1)
        else:
            return torch.cat([x,x_hat])
    
    def write(self, h_dec_out):
        w_t = self.write_linear(h_dec_out)
        if self.atten==True:
            Fx, Fy, gamma = self.get_filter(h_dec_out)
            Fy = Fy.transpose(2,1)
            
            w_t = w_t.view(-1,self.N,self.N)
            w_t = Fy.bmm(w_t.bmm(Fx)).view(-1,self.B*self.A)
            w_t = w_t / gamma.view(-1,1).expand_as(w_t)
        return w_t
        
    def Q(self, h_enc_out):
        normal = torch.randn(self.batch_size,self.rep_dim).to(self.device)
        mu = self.mu_linear(h_enc_out)
        logsigma = self.sigma_linear(h_enc_out)
        sigma = logsigma.exp()
        return mu + normal*sigma, mu, logsigma, sigma
        
    def loss(self, x):
        c = self.forward(x)
        
        x_recons = self.sigmoid(c[-1])
        
        Lx = self.criterion(x_recons,x) * self.A * self.B
        Lz = 0
        
        kl_terms = [0] * self.T
        
        for t in range(self.T):
            mu_2 = self.mu[t] **2
            sigma_2 = self.sigma[t]**2
            logsigma = self.logsigma[t]
            kl_terms[t] = 0.5 * torch.sum(mu_2+sigma_2-2 * logsigma,1) - self.T * 0.5
            Lz += kl_terms[t]
        
        Lz = torch.mean(Lz)   
        loss = Lz + Lx
        return loss

    def get_filter(self, h_dec):
        h_dec = h_dec.to(self.device)
        
        output = self.read_linear(h_dec)
        gx, gy, logvar, logdelta, loggamma= output.split(1,1)
        var = logvar.exp().unsqueeze(-1)
        var = var.expand((self.batch_size, self.N, self.A))
        
        Gx = 0.5*(self.A+1)*(gx+1)
        Gy = 0.5*(self.B+1)*(gy+1)
        
        delta = (max(self.A, self.B)-1)/(self.N-1) * logdelta.exp()
           
        index = [i for i in range(self.N)]
        index = torch.Tensor(index).to(self.device)
        index = index.unsqueeze(0)
        mux = Gx + (index -self.N/2 -0.5) * delta
        muy = Gy + (index -self.N/2 -0.5) * delta
        
        mux = mux.unsqueeze(2)
        muy = muy.unsqueeze(2)
        
        Fx = torch.zeros((self.batch_size,self.N,self.A)).to(self.device)
        Fy = torch.zeros((self.batch_size,self.N,self.B)).to(self.device)
        
        self.a = [i for i in range(self.A)]
        self.a = torch.Tensor(self.a).to(self.device)
        self.a = self.a.unsqueeze(0)
        self.a = self.a.unsqueeze(1)
        self.a = self.a.expand((self.batch_size, self.N, self.A))
        
        
        Fx = torch.exp(-((self.a-mux)**2/(2*var)))
        Fy = torch.exp(-((self.a-muy)**2/(2*var)))
                
        Fx = Fx/(Fx.sum(-1,True).expand_as(Fx)+ self.eps)
        Fy = Fy/(Fy.sum(-1,True).expand_as(Fy) + self.eps)
        return Fx, Fy, loggamma.exp()
    
    def generate(self,batch_size=64):
        self.batch_size = batch_size
        h_dec_out = torch.zeros(self.batch_size,self.dec_size).to(self.device)
        dec_hidden = torch.zeros(self.batch_size, self.dec_size).to(self.device)
        
        c = torch.zeros(self.T, self.batch_size,self.A * self.B).to(self.device)
        
        for t in range(self.T):
            prev_c = torch.zeros(self.batch_size, self.A * self.B).to(self.device) if t == 0 else c[t - 1]
            z_t = torch.randn(self.batch_size,self.rep_dim).to(self.device)
            h_dec_out, dec_hidden = self.decoder(z_t, (h_dec_out, dec_hidden))
            c[t] = prev_c + self.write(h_dec_out)
        imgs = []
        for img in c:
            imgs.append(self.sigmoid(img).cpu().data.numpy())
        return np.array(imgs)
from torch import nn
import torch
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, input_dim=3600, h1_dim=1800, h2_dim=900, h3_dim=450, z_dim=225):
        # init
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h1_dim
        self.z_dim = z_dim

        # encoder
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, h3_dim)
        self.fc4 = nn.Linear(h3_dim, z_dim) # mu
        self.fc5 = nn.Linear(h3_dim, z_dim) # log_var

        
        # decoder
        self.fc6 = nn.Linear(z_dim, h3_dim)
        self.fc7 = nn.Linear(h3_dim, h2_dim)
        self.fc8 = nn.Linear(h2_dim, h1_dim)
        self.fc9 = nn.Linear(h1_dim, input_dim)
        
        
    def forward(self, x):
        batch_size = x.shape[0]  # 每一批含有的样本的个数
        # flatten  [b, batch_size, 1, 28, 28] => [b, batch_size, 784]
        # tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。view不会修改自身的数据，
        # 返回的新tensor与原tensor共享内存，即更改一个，另一个也随之改变。
        x = x.view(batch_size, self.input_dim)  # 一行代表一个样本

        # encoder
        mu, log_var = self.encode(x)
        # reparameterization trick
        sampled_z = self.reparameterization(mu, log_var)
        # decoder
        x_hat = self.decode(sampled_z)
        # reshape
        x_hat = x_hat.view(batch_size, 1, 1, 3600)
        return x_hat, mu, log_var

    def encode(self, x):

        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        
        mu = self.fc4(h3)
        log_var = self.fc5(h3)

        return mu, log_var

    
    def reparameterization(self, mu, log_var):
       
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # 这里的“*”是点乘的意思

    
    def decode(self, z):
        
        h3 = F.relu(self.fc6(z))
        h2 = F.relu(self.fc7(h3))
        h1 = F.relu(self.fc8(h2))
        
        x_hat = torch.sigmoid(self.fc9(h1))  
        return x_hat

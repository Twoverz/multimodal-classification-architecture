import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, p,dropout = 0.):
        super().__init__()
        self.p = p
        self.cov = nn.Conv2d(in_channels=512, out_channels=4, kernel_size=(1, 1), stride=(1, 1))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        img_size = x.shape[2]
        x = self.cov(x)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p, p2=self.p)
        x = self.net(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p, p2=self.p, h=(img_size // self.p),
                      c=(x.shape[2] // self.p ** 2))
        return x


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(2, 512, 15, 15))#16 512 15 15
    #model = HSACon()
    model = FeedForward(dim=900,hidden_dim=1024,p=15)
    y = model(x)
    print('Output shape:', y.shape)

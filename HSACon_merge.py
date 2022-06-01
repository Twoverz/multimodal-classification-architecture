import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head *heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        #print(x.shape)torch.Size([2, 50, 1024])
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)#最后一维时通道数
        # 线性变换改变维度，chunk沿着指定轴最后一维分3块
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # einsum对张量的求和运算，默认成对出现的下标为求和下标
        mask_value = -torch.finfo(dots.dtype).max
        # finfo 表示浮点的数值属性的对象
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        #张量求和
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class HSACon(nn.Module):
    #dim决定了最后自注意模块输出的通道数  注意与out_channle保持一致，便于通道融合
    def __init__(self, in_channle=3, out_channle=1, patch_size=32, dim=1024, heads=8, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        patch_dim = channels * patch_size ** 2
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.att = Attention(dim, heads, dim_head, dropout)
        self.to_cls_token = nn.Identity()
        self.con = nn.Conv2d(in_channle, out_channle, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, mask=None):
        y = self.con(x)
        img_size = x.shape[2]
        p = self.patch_size
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        x = self.att(x)#torch.Size([2, 49, 1024])
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=p, p2=p, h=(img_size//p), c=(x.shape[2]//p**2))
        x = self.to_cls_token(x)
        #print(y.shape,x.shape)
        output = y+x    #通道融合  也可以拼接
        return output

class FeedForward(nn.Module):
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
        #print(x.shape)
        img_size = x.shape[2]
        x = self.cov(x)#torch.Size([2, 4, 15, 15])
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p, p2=self.p)
        x = self.net(x)#torch.Size([2, 1, 450])
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p, p2=self.p, h=(img_size // self.p),
                      c=(x.shape[2] // self.p ** 2))
        return x

class FeedForward_vgg(nn.Module):
    def __init__(self, dim, hidden_dim, outdim,dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, outdim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = self.net(x)#torch.Size([2, 1, 450])
        return x

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(2, 512, 15, 15))#16 512 15 15
    #model = HSACon()
    model = FeedForward(dim=900,hidden_dim=1024,p=15)
    y = model(x)
    print('Output shape:', y.shape)
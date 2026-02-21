import numpy as np, sys, torch, argparse, torchvision, matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import networks, math
from math import sqrt

#following https://medium.com/@heyamit10/implement-self-attention-and-cross-attention-in-pytorch-cfe17ab0b3ee ...
def scaled_DPA(Q,K,V):
    dk = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2,-1)) / torch.sqrt(torch.tensor(dk))
    attention = F.softmax(scores, dim=-1)

    return torch.matmul(attention, V)

#positional encoder taken from
#https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
#with minor edits
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.transpose(0,1)

        x = x + self.pe[:x.size(0)]
        return self.dropout(x.transpose(0,1))
    
class MNIST_patch_embed(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        #going with square image and square patches and a single channel
        assert img_size % patch_size == 0, "image side length % patch side length must be 0"
        self.n_patches = ( img_size // patch_size )**2 
        #convolution will yield (N, n_hidden, n_patches, patch_size, patch_size) -- n_hidden is embedding dimension
        self.conv = nn.LazyConv2d(embed_dim, kernel_size = patch_size, stride = patch_size)

    def forward(self, x):
        #want to yield (N=batch_size, n_patches, embed_dim)
        return self.conv(x).flatten(-2,-1).transpose(-2,-1)

class mnist_encoder(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = MH_layer(embed_dim, n_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )

    def forward(self, x):
        out = x + self.mha(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out

class mnist_transformer(nn.Module):
    def __init__(self, embed_dim, n_heads, img_size, patch_size, n_coders):
        super().__init__()
        
        assert img_size % patch_size == 0, "patch size must fit evenly into img size"
        assert embed_dim % n_heads == 0, "number of heads must fit evenly into embed dimension"

        self.n_patches = ( img_size // patch_size ) **2
        self.max_seq_length = self.n_patches

        self.patch_embed = MNIST_patch_embed(img_size, patch_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout = 0.1, max_len = self.max_seq_length) 
        self.transformer = nn.Sequential(*[mnist_encoder(embed_dim, n_heads) for _ in range(n_coders)])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 10)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        # print("after patch embed ", x.shape)
        x = self.positional_encoding(x)#(x.transpose(0,1)).transpose(0,1)
        # print(f"after positional encoding {x.shape}")
        x = self.transformer(x)
        # print(f"after transformer {x.shape}")
        x = self.classifier(x[:,0])
        return x

#single_attention_head assumes dq=dk=dv and target length = source length
class single_attention_head(nn.Module):
    def __init__(self, d_embed, d_h, res = False):
        super().__init__()
        self.head_size = d_h
        self.res = res
        
        self.Wq = nn.Linear(d_embed, self.head_size)
        self.Wk = nn.Linear(d_embed, self.head_size)
        self.Wv = nn.Linear(d_embed, self.head_size)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        y = scaled_DPA(Q,K,V)
        if self.res is True: y = x + y
        return y

class MH_layer(nn.Module):
    def __init__(self, d_embed, num_heads, mlp_mult = 4, res = False):
        super().__init__()

        assert d_embed % num_heads == 0, "d_embed must be divisible by num_heads"

        self.head_dim = int(d_embed / num_heads)
        self.num_heads = num_heads
        self.res = res #are we residual-ing?

        self.heads = nn.ModuleList([single_attention_head(d_embed, self.head_dim, res = False) for _ in range(num_heads)])
        self.Wo = nn.Linear(d_embed, d_embed)

    def forward(self, x):  
        #each head should multiply seq x embed_dim inputs with embed_dim x head_dim and output seq x head_dim
        out = torch.cat([head(x) for head in self.heads], dim=-1)  #concatenate each head to get seq x (head_dim * num_heads = d_embed)
        out = self.Wo(out) #each head is separately weighted
        if self.res is True: out = x + out #x is seq x embed_dim, same as input
        return out

def main():
    N_batch, img_size, patch_size, embed_dim, n_heads = 10, 28, 4, 64, 8
    n_chan = 1 #doing one channel for MNIST


    fake_images = torch.randn(N_batch, n_chan, img_size, img_size)

    print("fake images shape ", fake_images.shape)
    patch_emb = MNIST_patch_embed(img_size, patch_size, embed_dim)
    embed = patch_emb(fake_images)
    print("patch embedding shape ", embed.shape)
    posish = PositionalEncoding(embed_dim, dropout = 0.1, max_len = 28*28+1)
    print("posish shape ", posish(embed).shape)
    # print ("patch embed shape: ", embed.shape) #should be 7*7 patches with 64 channels each 

    # sa_head = single_attention_head(embed_dim, embed_dim, res = False)

    # print(sa_head(posish(embed)).shape)
    # mh_head = MH_layer(embed_dim, 8)
    # print(mh_head(posish(embed)).shape)

    transformer = mnist_transformer(embed_dim, n_heads, img_size, patch_size, 16) #embed_dim, n_heads, img_size, patch_size, n_coders

    print("transformer of fake images ", transformer(fake_images).shape)
    # #Example shapes
    # seq1 = 10
    # seq2 = 15
    # embed_dim = 16
    # dkq = 16
    # dv = 32

    # # Input sequences
    # x1 = torch.randn(seq1, embed_dim)
    # x2 = torch.randn(seq2, embed_dim)

    # # Project to attention space
    # Wq = nn.Linear(embed_dim, dkq)
    # Wk = nn.Linear(embed_dim, dkq)
    # Wv = nn.Linear(embed_dim, dv)

    # Q = Wq(x1)
    # K = Wk(x2)
    # V = Wv(x2)

    # output = scaled_DPA(Q,K,V)  # [4, 20, 64]

    # print(output.shape)

if __name__ == '__main__':
    main()
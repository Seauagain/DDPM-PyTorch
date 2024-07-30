# system-level import 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn import init
import math 

# custom-level import 
from .activation import SiLU

class PositionalEmbedding(nn.Module):
    r"""compute the positional embedding of timesteps."""

    def __init__(self, T, d_model, dim, args):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            SiLU(),
            nn.Linear(dim, dim),
        )
        self.std_rate = args.std_rate
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # init.xavier_uniform_(module.weight)
                init.normal_(module.weight, mean = 0, std = module.weight.size(1)**(-self.std_rate))
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    r"""Downsamples a given tensor by a factor of 2. Uses strided convolution. Assumes even height and width."""
    
    def __init__(self, in_channels, args):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
        self.std_rate = args.std_rate
        self.initialize()

    def initialize(self):
        # init.xavier_uniform_(self.conv.weight)
        init.normal_(self.conv.weight, mean=0, std=self.conv.weight.size(1)**(-self.std_rate))
        init.zeros_(self.conv.bias)

    def forward(self, x, time_emb):
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    r"""Upsamples a given tensor by a factor of 2. Uses resize convolution to avoid checkerboard artifacts."""

    def __init__(self, in_channels, args):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.std_rate = args.std_rate
        self.initialize()

    def initialize(self):
        init.normal_(self.conv.weight, mean = 0, std = self.conv.weight.size(1)**(-self.std_rate))
        # init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    r"""Applies QKV self-attention with a residual connection."""

    def __init__(self, in_channels, args):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.proj_q = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)

        self.std_rate = args.std_rate
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            # init.xavier_uniform_(module.weight)
            init.normal_(module.weight, mean=0, std=module.weight.size(1)**(-self.std_rate))
            init.zeros_(module.bias)
        # init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, tdim, dropout, attn=False, args={}):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            SiLU(),
            nn.Linear(tdim, out_channels),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttentionBlock(out_channels, args)
        else:
            self.attn = nn.Identity()

        self.std_rate = args.std_rate
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # init.xavier_uniform_(module.weight)
                init.normal_(module.weight, mean=0, std=module.weight.size(1)**(-self.std_rate))
                init.zeros_(module.bias)

        init.normal_(self.block2[-1].weight, mean=0, std=self.block2[-1].weight.size(1)**(-self.std_rate))
        # init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.temb_proj(time_emb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    """UNet model used to estimate noise."""

    def __init__(self, args):
        super().__init__()
        T = args.T 
        ch = args.channel
        ch_mult = args.channel_mult
        attn = args.attn
        num_res_blocks = args.num_res_blocks
        if args.state == "train":
            dropout = args.dropout
        elif args.state == "eval":
            dropout = 0
        else:
            raise ValueError(f"expect args.state be 'train' or 'eval' but got '{args.state}'")

        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = PositionalEmbedding(T, ch, tdim, args)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_channels = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResidualBlock(
                    in_channels=now_ch, out_channels=out_channels, tdim=tdim,
                    dropout=dropout, attn=(i in attn), args=args))
                now_ch = out_channels
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch, args))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResidualBlock(now_ch, now_ch, tdim, dropout, attn=True, args=args),
            ResidualBlock(now_ch, now_ch, tdim, dropout, attn=False, args=args),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_channels = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResidualBlock(
                    in_channels=chs.pop() + now_ch, out_channels=out_channels, tdim=tdim,
                    dropout=dropout, attn=(i in attn), args=args))
                now_ch = out_channels
            if i != 0:
                self.upblocks.append(UpSample(now_ch, args))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            SiLU(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )

        self.std_rate = args.std_rate
        self.initialize()

    def initialize(self):
        # init.xavier_uniform_(self.head.weight)
        init.normal_(self.head.weight, mean=0, std=self.head.weight.size(1)**(-self.std_rate))
        init.zeros_(self.head.bias)
        # init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.normal_(self.tail[-1].weight, mean=0, std=self.tail[-1].weight.size(1)**(-self.std_rate))
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
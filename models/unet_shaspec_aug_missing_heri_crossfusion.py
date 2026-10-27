import torch
from torch import nn
from torchvision.ops import StochasticDepth
import sys
# sys.path.append("..")
# print("sys.path---------",sys.path)

from .utils.helpers import as_tuple, prod, wrap_class 
# from utils.helpers import as_tuple, prod, wrap_class # for test
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.distributions import Uniform

gaussian_att = False
visualization = False
tsne_points = []
tsne_labels = []
tsne_colors = []

def plot_embedding_2D(data, label, colors, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # fig = plt.figure()
    fig = plt.figure(figsize=(6.5, 6.5))

    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(colors[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

class Linear(nn.Module):
    """"Linear layer for channels-first inputs."""

    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype=None,
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=2)
        self.linear = nn.Conv1d(
            in_features,
            out_features,
            kernel_size=1,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        shape = x.shape
        out = self.flatten(x)
        out = self.linear(out)
        out = out.reshape(shape[0], -1, *shape[2:])
        return out


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x

def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class DoubleConv(nn.Module):
    """(Conv -- Drop -- Norm -- Act) ** 2."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        conv = wrap_class(conv)
        drop = wrap_class(drop)
        norm = wrap_class(norm)
        act = wrap_class(act)

        self.block1 = nn.Sequential(
            conv(in_channels, mid_channels, stride=stride),
            drop(),
            norm(mid_channels),
            act(),
        )

        self.block2 = nn.Sequential(
            conv(mid_channels, out_channels, stride=1),
            drop(),
            norm(out_channels),
            act(),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out


class BasicBlock(nn.Module):
    """Basic ResNet block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        conv1 = wrap_class(conv)
        conv2 = wrap_class(conv)
        drop = wrap_class(drop)
        norm = wrap_class(norm)
        act = wrap_class(act)

        self.conv1 = conv1(in_channels, mid_channels, stride=stride)
        self.drop1 = drop()
        self.norm1 = norm(mid_channels)
        self.conv2 = conv2(mid_channels, out_channels)
        self.drop2 = drop()
        self.norm2 = norm(out_channels)
        self.act = act()

        if prod(as_tuple(stride)) != 1 or in_channels != out_channels:
            self.shortcut = conv[0](
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.drop1(out)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.drop2(out)
        out = self.norm2(out)

        out += shortcut

        out = self.act(out)

        return out


class BottleneckBlock(nn.Module):
    """ResNet bottleneck block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.5,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = int(in_channels * alpha)

        conv2 = wrap_class(conv)
        drop = wrap_class(drop)
        norm = wrap_class(norm)
        act = wrap_class(act)

        self.conv1 = conv[0](in_channels, mid_channels, kernel_size=1)
        self.drop1 = drop()
        self.norm1 = norm(mid_channels)
        self.conv2 = conv2(mid_channels, mid_channels, stride=stride)
        self.drop2 = drop()
        self.norm2 = norm(mid_channels)
        self.conv3 = conv[0](mid_channels, out_channels, kernel_size=1)
        self.drop3 = drop()
        self.norm3 = norm(mid_channels)
        self.act = act()

        if prod(as_tuple(stride)) != 1 or in_channels != out_channels:
            self.shortcut = conv[0](
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.drop1(out)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.drop2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.drop3(out)
        out = self.norm3(out)

        out += shortcut
        out = self.act(out)

        return out


class PreActivationBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        conv1 = wrap_class(conv)
        conv2 = wrap_class(conv)
        drop = wrap_class(drop)
        norm = wrap_class(norm)
        act = wrap_class(act)

        self.norm1 = norm(in_channels)
        self.act = act()
        self.conv1 = conv1(in_channels, mid_channels, stride=stride)
        self.drop1 = drop()
        self.norm2 = norm(mid_channels)
        self.conv2 = conv2(mid_channels, out_channels)
        self.drop2 = drop()

        if prod(as_tuple(stride)) != 1 or in_channels != out_channels:
            self.shortcut = conv[0](
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x):
        out = self.norm1(x)
        out = self.act(out)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.drop1(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.drop2(out)

        out += shortcut
        return out


class CovNeXtBlock(nn.Module):
    """ ConvNeXt Block"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mlp_ratio=4,
        stochastic_depth_prob=0.0,
        layer_scale=1e-6,
        conv=(nn.Conv3d, {"kernel_size": 7, "padding": 3}),
        norm=nn.LayerNorm,
        act=nn.GELU,
        **kwargs,
    ):
        super().__init__()

        conv = wrap_class(conv)
        norm = wrap_class(norm)
        act = wrap_class(act)

        if in_channels != out_channels:
            self.adapter = Linear(in_channels, out_channels, bias=False)

        hidden_channels = int(mlp_ratio * in_channels)
        self.dwconv = conv(out_channels, out_channels, groups=out_channels)
        self.norm = norm(out_channels)
        self.pwconv1 = nn.Linear(out_channels, hidden_channels)
        self.act = act()
        self.pwconv2 = nn.Linear(hidden_channels, out_channels)
        self.gamma = (
            nn.Parameter(layer_scale * torch.ones(out_channels))
            if layer_scale > 0
            else None
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x):
        x = self.adapter(x) if hasattr(self, "adapter") else x
        out = self.dwconv(x)
        out = torch.einsum("b c ... -> b ... c", out)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        if self.gamma is not None:
            out = self.gamma * out
        out = torch.einsum("b ... c -> b c ...", out)
        out = self.stochastic_depth(out)
        out += x
        return x


class Same(object):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def __getitem__(self, *args, **kwargs):
        return self.block


class UNetEncoderBlock(nn.Module):
    """U-Net encoder block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        depth=1,
        stride=2,
        downsample=(nn.Conv3d, {"kernel_size": 2}),
        block=Same(PreActivationBlock),
    ):
        super().__init__()
        if prod(as_tuple(stride)) == 1:
            wrapped_block = wrap_class(block[0])
            self.blocks = [wrapped_block(in_channels, out_channels)]

        elif downsample == "on-block":
            wrapped_block = wrap_class(block[0])
            self.blocks = [
                wrapped_block(in_channels, out_channels, stride=stride)
            ]

        else:
            downsample = wrap_class(downsample)
            wrapped_block = wrap_class(block[0])
            self.blocks = [
                downsample(in_channels, out_channels, stride=stride),
                wrapped_block(out_channels, out_channels),
            ]

        for i in range(1, depth):
            wrapped_block = wrap_class(block[i])
            self.blocks.append(wrapped_block(out_channels, out_channels))

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


def layer_iterator(depth_list):
    for layer, depth in enumerate(depth_list):
        for sublayer in range(depth):
            yield layer, sublayer


class UNetEncoder(nn.Module):
    """U-Net encoder."""

    def __init__(
        self,
        in_channels,
        depth=(1, 1, 1, 1, 1),
        width=(32, 64, 128, 256, 512),
        strides=(1, 2, 2, 2, 2),
        downsample=None,
        block=Same(PreActivationBlock),
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UNetEncoderBlock(
                    in_channels,
                    width[0],
                    depth[0],
                    strides[0],
                    downsample,
                    {
                        j: block[i, j]
                        for i, j in layer_iterator(depth)
                        if i == 0
                    },
                ),
            ]
        )
        for i in range(len(width) - 1):
            self.blocks.append(
                UNetEncoderBlock(
                    width[i],
                    width[i + 1],
                    depth[i + 1],
                    strides[i + 1],
                    downsample,
                    {
                        j: block[h, j]
                        for h, j in layer_iterator(depth)
                        if h == i + 1
                    },
                )
            )

    def forward(self, x):
        out = [self.blocks[0](x)]
        for block in self.blocks[1:]:
            out.append(block(out[-1]))

        return out


class UNetDecoderBlock(nn.Module):
    """U-Net decoder block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        depth=1,
        stride=2,
        upsample=(nn.ConvTranspose3d, {"kernel_size": 2}),
        block=Same(PreActivationBlock),
    ):
        super().__init__()
        upsample = wrap_class(upsample)
        self.upsample = upsample(in_channels, out_channels, stride=stride)

        wrapped_block = wrap_class(block[0])
        self.blocks = [wrapped_block(2 * out_channels, out_channels)]
        for i in range(1, depth):
            wrapped_block = wrap_class(block[i])
            self.blocks.append(wrapped_block(out_channels, out_channels))

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.blocks(x)
        return x


class UNetDecoder(nn.Module):
    """U-Net decoder."""

    def __init__(
        self,
        in_channels,
        depth=(1, 1, 1, 1),
        width=(256, 128, 64, 32),
        strides=(2, 2, 2, 2),
        upsample=None,
        block=Same(PreActivationBlock),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.blocks = nn.ModuleList(
            [
                UNetDecoderBlock(
                    in_channels,
                    width[0],
                    depth[0],
                    strides[0],
                    upsample,
                    {
                        j: block[i, j]
                        for i, j in layer_iterator(depth)
                        if i == 0
                    },
                )
            ]
        )
        for i in range(len(width) - 1):
            self.blocks.append(
                UNetDecoderBlock(
                    width[i],
                    width[i + 1],
                    depth[i + 1],
                    strides[i + 1],
                    upsample,
                    {
                        j: block[h, j]
                        for h, j in layer_iterator(depth)
                        if h == i + 1
                    },
                )
            )

    def forward(self, x):
        out = x
        for i, block in enumerate(self.blocks):
            i0 = -1 - i
            i1 = -2 - i
            out[i1] = block(out[i0], out[i1])
            # print(f"Decoder block {i} output shape:", out[i1].shape)

        return out



class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                 groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4,
                                                                                                                keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1,
              bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)


class CompositionalLayer(nn.Module):

    def __init__(self, weight_std=False, dim=512, normalization_sign=False):
        super().__init__()
        self.normalization = normalization_sign
        # self.conv = conv3x3x3(32 * 2, 32, kernel_size=1, bias=False, weight_std=weight_std)
        # self.conv = conv3x3x3(32 * 2, 32, kernel_size=3, padding=1, bias=False, weight_std=weight_std)
        # self.conv = conv3x3x3(512 * 2, 512, kernel_size=3, padding=1, bias=False, weight_std=weight_std)
        self.conv = conv3x3x3(dim * 2, dim, kernel_size=3, padding=1, bias=False, weight_std=weight_std)

    def forward(self, f1, f2):
        """
        :param f1: shared-modality fts
        :param f2: specific-modality fts
        :return:
        """
        if self.normalization:
            f1_n = F.normalize(f1, dim=1)
            f2_n = F.normalize(f2, dim=1)
            residual = torch.cat((f1_n, f2_n), 1)
        else:
            residual = torch.cat((f1, f2), 1)

        #### compose two modalities by residual learning
        residual = self.conv(residual)
        features = f1 + residual  # other modality + residual (default)

        return features

class UNet(nn.Module):
    """Generic U-Net architecture."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_dims=3,
        stem_width=None,
        encoder_depth=(1, 1, 1, 1, 1),
        encoder_width=(32, 64, 128, 256, 512),
        strides=(1, 2, 2, 2, 2),
        decoder_depth=(1, 1, 1, 1),
        stem=None,
        downsample=None,
        block=None,
        upsample=None,
        head=None,
        num_deep_supr=1,
    ):
        super().__init__()
        conv = getattr(nn, f"Conv{spatial_dims}d")
        tconv = getattr(nn, f"ConvTranspose{spatial_dims}d")
        if stem is None:
            stem = (conv, {"kernel_size": 3, "padding": 1, "bias": False})

        if downsample is None:
            downsample = (conv, {"kernel_size": 2})

        if block is None:
            block = (
                PreActivationBlock,
                {"conv": (conv, {"kernel_size": 3, "padding": 1})},
            )

            # block=(
            #     BasicBlock,
            #     {
            #         "conv": (nn.Conv3d, {"kernel_size": 3, "padding": 1}),
            #         "norm": (nn.GroupNorm, (8,)),
            #         "act": nn.LeakyReLU,
            #     },
            # )



            block = Same(block)

        if upsample is None:
            upsample = (tconv, {"kernel_size": 2})

        if head is None:
            head = (conv, {"kernel_size": 1})

        stem = wrap_class(stem)
        head = wrap_class(head)

        if stem_width is None:
            stem_width = encoder_width[0]

        self.stem_sha = stem(in_channels, stem_width)
        self.stem_spec_m1 = stem(in_channels, stem_width)
        self.stem_spec_m2 = stem(in_channels, stem_width)


        self.encoder_sha = UNetEncoder(
            stem_width,
            encoder_depth,
            encoder_width,
            strides,
            downsample,
            block,
        )
        self.decoder_sha = UNetDecoder(
            encoder_width[-1],
            decoder_depth,
            encoder_width[-2::-1],
            strides[:0:-1],
            upsample,
            {
                (i, j): block[i + len(encoder_depth), j]
                for i, j in layer_iterator(decoder_depth)
            },
        )

        # self.decoder_sha_fused = UNetDecoder(
        #     encoder_width[-1],
        #     decoder_depth,
        #     encoder_width[-2::-1],
        #     strides[:0:-1],
        #     upsample,
        #     {
        #         (i, j): block[i + len(encoder_depth), j]
        #         for i, j in layer_iterator(decoder_depth)
        #     },
        # )

        self.decoder_sha_fused = UNetDecoder(
            encoder_width[-1] * 2,
            decoder_depth,
            [x * 2 for x in encoder_width[-2::-1]],
            strides[:0:-1],
            upsample,
            {
                (i, j): block[i + len(encoder_depth), j]
                for i, j in layer_iterator(decoder_depth)
            },
        )



        self.encoder_spec_m1 = UNetEncoder(
            stem_width,
            encoder_depth,
            encoder_width,
            strides,
            downsample,
            block,
        )
        self.decoder_spec = UNetDecoder(
            encoder_width[-1],
            decoder_depth,
            encoder_width[-2::-1],
            strides[:0:-1],
            upsample,
            {
                (i, j): block[i + len(encoder_depth), j]
                for i, j in layer_iterator(decoder_depth)
            },
        )

        self.encoder_spec_m2 = UNetEncoder(
            stem_width,
            encoder_depth,
            encoder_width,
            strides,
            downsample,
            block,
        )
        # self.decoder_spec_m2 = UNetDecoder(
        #     encoder_width[-1],
        #     decoder_depth,
        #     encoder_width[-2::-1],
        #     strides[:0:-1],
        #     upsample,
        #     {
        #         (i, j): block[i + len(encoder_depth), j]
        #         for i, j in layer_iterator(decoder_depth)
        #     },
        # )


        self.num_deep_supr = num_deep_supr
        self.heads_shared = nn.ModuleList([])
        self.heads_shared_aug = nn.ModuleList([])

        for j in range(num_deep_supr):
            self.heads_shared.append(head(encoder_width[j]*2, out_channels))
            self.heads_shared_aug.append(head(encoder_width[j]*2, out_channels))


        self.heads_shared_m1 = nn.ModuleList([])
        self.heads_shared_m2 = nn.ModuleList([])
        for j in range(num_deep_supr):
            self.heads_shared_m1.append(head(encoder_width[j], out_channels))
            self.heads_shared_m2.append(head(encoder_width[j], out_channels))



        self.heads_spec_m1 = nn.ModuleList([])
        self.heads_spec_m2 = nn.ModuleList([])  

        for j in range(num_deep_supr):
            self.heads_spec_m1.append(head(encoder_width[j], out_channels))
            self.heads_spec_m2.append(head(encoder_width[j], out_channels)) 



        # self.heads_shared_pseodo = nn.ModuleList([])
        # for j in range(num_deep_supr):
        #     self.heads_shared_pseudo.append(head(encoder_width[j]*2, out_channels))

        # self.heads_spec_m1_pseudo = nn.ModuleList([])
        # self.heads_spec_m2_pseudo = nn.ModuleList([])  

        # for j in range(num_deep_supr):
        #     self.heads_spec_m1_pseudo.append(head(encoder_width[j], out_channels))
        #     self.heads_spec_m2_pseudo.append(head(encoder_width[j], out_channels)) 
    


        self.compos_layer = CompositionalLayer(weight_std=False, dim =encoder_width[-1] )  # a shared compos layer
        self.compos_layer_shared = CompositionalLayer(weight_std=False, dim =encoder_width[-1])  # a shared compos layer

        self.multihead_attn = nn.MultiheadAttention(embed_dim=encoder_width[-1], num_heads=4, batch_first=True)
      
        self.multihead_attn_m12m2 = nn.MultiheadAttention(embed_dim=encoder_width[-1], num_heads=4, batch_first=True)
        self.multihead_attn_m22m1 = nn.MultiheadAttention(embed_dim=encoder_width[-1], num_heads=4, batch_first=True)

    def forward_features(self, images, mode_split):
        N, C, D, W, H = images.shape

        # _images = images.clone()  
        x_m1 = images[:, 0, :, :, :].unsqueeze(dim=1)
        x_m2 = images[:, 1, :, :, :].unsqueeze(dim=1)

        x_m1_shared = self.stem_sha(x_m1)
        shared_m1_ft = self.encoder_sha(x_m1_shared) #[torch.Size([1, 32, 64, 64, 64]), torch.Size([1, 64, 32, 32, 32]), torch.Size([1, 128, 16, 16, 16]), torch.Size([1, 256, 8, 8, 8]), torch.Size([1, 512, 4, 4, 4])]
        shared_m1_ft_final = shared_m1_ft[-1]

        x_m2_shared = self.stem_sha(x_m2)
        shared_m2_ft = self.encoder_sha(x_m2_shared)
        shared_m2_ft_final = shared_m2_ft[-1]

        # shared_m12_ft = [a + b for a, b in zip(shared_m1_ft, shared_m2_ft)]
        shared_m12_ft = [torch.cat([a, b], dim=1) for a, b in zip(shared_m1_ft, shared_m2_ft)]



        # Forward through spec_m1 encoder
        x_m1 = self.stem_spec_m1(x_m1)
        spec_m1_ft = self.encoder_spec_m1(x_m1)
        spec_m1_ft_final = spec_m1_ft[-1]  #[1, 512, 4, 4, 4]

        # Forward through spec_m2 encoder
        x_m2 = self.stem_spec_m2(x_m2)
        spec_m2_ft = self.encoder_spec_m2(x_m2)
        spec_m2_ft_final = spec_m2_ft[-1]  #[1, 512, 4, 4, 4]   



        if 0 in mode_split and 1 in mode_split:
            original_size = spec_m1_ft_final.size()
            flat_input_Q = spec_m1_ft_final.view(original_size[0], original_size[1],
                                            original_size[2] * original_size[3] * original_size[4]).permute(0, 2, 1) #(1,64, 512)
            flat_input_K = spec_m2_ft_final.view(original_size[0], original_size[1],
                                            original_size[2] * original_size[3] * original_size[4]).permute(0, 2, 1)
            m1_spec_enhance_ft, att_weights = self.multihead_attn_m22m1(flat_input_Q, flat_input_K, flat_input_K)
            m1_spec_enhance_ft = m1_spec_enhance_ft.permute(0, 2, 1).view(original_size[0], original_size[1],  original_size[2],  original_size[3],  original_size[4]) #[1, 512, 4, 4, 4]  


            flat_input_Q2 = spec_m2_ft_final.view(original_size[0], original_size[1],
                                            original_size[2] * original_size[3] * original_size[4]).permute(0, 2, 1) #(1,64, 512)
            flat_input_K2 = spec_m1_ft_final.view(original_size[0], original_size[1],
                                            original_size[2] * original_size[3] * original_size[4]).permute(0, 2, 1)
            m2_spec_enhance_ft, att_weights2 = self.multihead_attn_m12m2(flat_input_Q2, flat_input_K2, flat_input_K2)
            m2_spec_enhance_ft = m2_spec_enhance_ft.permute(0, 2, 1).view(original_size[0], original_size[1],  original_size[2],  original_size[3],  original_size[4]) #[1, 512, 4, 4, 4]  

        if (0 in mode_split) and (1 not in mode_split):
            m1_spec_enhance_ft = spec_m1_ft_final
            m2_spec_enhance_ft = spec_m2_ft_final

        if (1 in mode_split) and (0 not in mode_split):
            m2_spec_enhance_ft = spec_m2_ft_final
            m1_spec_enhance_ft = spec_m1_ft_final


        # Combine features
        if 0 in mode_split:
            m1_fused_ft = self.compos_layer(shared_m1_ft_final, m1_spec_enhance_ft) #[1, 512, 4, 4, 4]  dwi
        else:
            m1_fused_ft = shared_m2_ft_final
        if 1 in mode_split:
            m2_fused_ft = self.compos_layer(shared_m2_ft_final, m2_spec_enhance_ft) #[1, 512, 4, 4, 4]  
        else:
            m2_fused_ft = shared_m1_ft_final

        m12_fused_ft = torch.cat((m1_fused_ft, m2_fused_ft), dim=1)  #[1, 512*2, 4, 4, 4]
        # m12_fused_ft = self.compos_layer_shared(m1_fused_ft, m2_fused_ft) #[1, 512, 4, 4, 4]  


        # Update the shared feature with the fused feature
        shared_m12_ft[-1] = m12_fused_ft
        spec_m1_ft[-1] = m1_spec_enhance_ft
        spec_m2_ft[-1] = m2_spec_enhance_ft
    
        random_number = random.random()
        if random_number > 0.5:
            shared_m12_ft_aug = [FeatureDropout(i) for i in shared_m12_ft]
        else:
            shared_m12_ft_aug = [FeatureNoise()(i) for i in shared_m12_ft]

        # shared_m12_ft_aug = [nn.Dropout3d(0.5)(feat) for feat in _shared_layers]

        
        # Decoder 
        if self.training:
            out_shared_m1 = self.decoder_sha(shared_m1_ft)
            out_shared_m2 = self.decoder_sha(shared_m2_ft)

            out_shared = self.decoder_sha_fused(shared_m12_ft)
           # print("Decoder output shapes:", [o.shape for o in out_shared]) #[1, 64, 64, 64, 64]), torch.Size([1, 128, 32, 32, 32]), torch.Size([1, 256, 16, 16, 16]), torch.Size([1, 512, 8, 8, 8]), torch.Size([1, 1024, 4, 4, 4])]

            out_shared_aug = self.decoder_sha_fused(shared_m12_ft_aug)

            out_spec_m1 = self.decoder_spec(spec_m1_ft)
            out_spec_m2 = self.decoder_spec(spec_m2_ft)

        else:
            # print("val------------")
            out_shared = self.decoder_sha_fused(shared_m12_ft)

            out_shared_m1 = out_shared
            out_shared_m2 = out_shared
            out_shared_aug = out_shared
            out_spec_m1 = out_shared
            out_spec_m2 = out_shared


            # out_shared_m1 = self.decoder_sha(shared_m1_ft)
            # out_shared_m2 = self.decoder_sha(shared_m2_ft)
            
            # out_shared = self.decoder_sha_fused(shared_m12_ft)
            # out_shared_aug = self.decoder_sha_fused(shared_m12_ft_aug)

            # out_spec_m1 = self.decoder_spec(spec_m1_ft)
            # out_spec_m2 = self.decoder_spec(spec_m2_ft)


        # return out
        return out_shared, out_shared_aug, out_shared_m1, out_shared_m2, out_spec_m1, out_spec_m2, shared_m1_ft_final, shared_m2_ft_final, spec_m1_ft_final, spec_m2_ft_final   

    def forward(self, image, mask_indices=None, val=False, mode='0,1'):

        # mode = '2,3'  # debug
        # random choose modes instead of specifying
        # training or mask_indices not specified       
        if not val or mask_indices is None:
            if mode == 'random':
                mode = '0,1'

                mode_split = mode.split(',')
                mode_split = list(map(int, mode_split))

                mode_num = random.randint(1, 2)
                mode_split = random.sample(mode_split, mode_num)
            else:
                mode_split = mode.split(',')
                mode_split = list(map(int, mode_split))
        # validation/test during training
        else:
            mode_split = mask_indices.split(',')
            mode_split = list(map(int, mode_split))

        # delete missing modalities
        _images = image.clone() 
        for m in [0, 1]:
            if m not in mode_split:
                _images[:, m, :, :, :] = 0

        out_shared, out_shared_aug, out_shared_m1, out_shared_m2,  out_spec_m1, out_spec_m2, shared_m1_ft_final, shared_m2_ft_final, spec_m1_ft_final, spec_m2_ft_final = self.forward_features(_images, mode_split)
       


          # visualization of shared-specific fts here
        if visualization:
            # collecting enough data points
            # tsne_2D = TSNE(n_components=2, random_state=0)
            tsne_2D = TSNE(n_components=2, perplexity=200, init='pca', random_state=0)
            vis_ft = torch.cat([shared_m1_ft_final, shared_m2_ft_final,
                                spec_m1_ft_final, spec_m2_ft_final])
            # print("vis_ft shape:", vis_ft.shape)  # torch.Size([4, 512, 4, 4, 4])
          
            vis_ft = vis_ft.view(vis_ft.shape[0], -1)
            tsne_points.append(vis_ft.cpu().detach().numpy())
            tsne_colors.append([0, 1, 7, 8])
            # tsne_labels.append([0, 1, 2, 3, 4, 5, 6, 7])
            tsne_labels.append(['x', 'x', 'o', 'o'])
            print(f"Collected {len(tsne_points)} sets of features for t-SNE visualization.") 
            if len(tsne_points) == 600:  # actual visualization
                vis_tsne_points = np.concatenate(tsne_points)
                print("vis_tsne_points shape------------:", vis_tsne_points.shape)  # (4*len(tsne_points) , 512*4*4*4)
                np.save('tsne_points.npy', vis_tsne_points)
                vis_tsne_labels = np.concatenate(tsne_labels)
                vis_tsne_colors = np.concatenate(tsne_colors)
                vis_tsne_2D = tsne_2D.fit_transform(vis_tsne_points)
                # print("tsne_2D shape:", vis_tsne_2D.shape)  # (32, 2)
                tsne_fig = plot_embedding_2D(vis_tsne_2D, vis_tsne_labels, vis_tsne_colors, 't-SNE of Features')
                # plt.show()
                plt.savefig('tsne.png')
                quit()

        # if visualization:
        #     shared_m1_ft_final_vis = shared_m1_ft_final.view(shared_m1_ft_final.shape[0], -1).cpu().detach().numpy()
        #     shared_m2_ft_final_vis = shared_m2_ft_final.view(shared_m2_ft_final.shape[0], -1).cpu().detach().numpy()
        #     spec_m1_ft_final_vis = spec_m1_ft_final.view(spec_m1_ft_final.shape[0], -1).cpu().detach().numpy()
        #     spec_m2_ft_final_vis = spec_m2_ft_final.view(spec_m2_ft_final.shape[0], -1).cpu().detach().numpy()
        #     print(shared_m1_ft_final_vis.shape)  # (1, 512*4*4*4)


        pred_shared = []
        pred_shared_aug = []
        
        pred_shared_m1 = []
        pred_shared_m2 = []

        pred_spec_m1 = []
        pred_spec_m2 = []

        if self.training:
            for j, head in enumerate(self.heads_shared):
                pred_shared.append(head(out_shared[j]))
            for j, head in enumerate(self.heads_shared_aug):  # not share head
                pred_shared_aug.append(head(out_shared_aug[j]))


            for j, head in enumerate(self.heads_shared_m1):
                pred_shared_m1.append(head(out_shared_m1[j]))
            for j, head in enumerate(self.heads_shared_m2):
                pred_shared_m2.append(head(out_shared_m2[j])) 


            for j, head in enumerate(self.heads_spec_m1):
                pred_spec_m1.append(head(out_spec_m1[j]))
            for j, head in enumerate(self.heads_spec_m2):
                pred_spec_m2.append(head(out_spec_m2[j]))

    


            return {
                "pred_shared": pred_shared,
                "pred_shared_aug": pred_shared_aug,

                "pred_shared_m1": pred_shared_m1,
                "pred_shared_m2": pred_shared_m2,

                "pred_spec_m1": pred_spec_m1,
                "pred_spec_m2": pred_spec_m2,

                "shared_m1_ft": shared_m1_ft_final,
                "shared_m2_ft": shared_m2_ft_final,
                "spec_m1_ft": spec_m1_ft_final,
                "spec_m2_ft": spec_m2_ft_final,
            }
        else:
             for j, head in enumerate(self.heads_shared):
                pred_shared.append(head(out_shared[j]))

             return {
                "pred_shared": pred_shared,

                "shared_m1_ft": shared_m1_ft_final,
                "shared_m2_ft": shared_m2_ft_final,
                "spec_m1_ft": spec_m1_ft_final,
                "spec_m2_ft": spec_m2_ft_final,
            }

def test_unet_forward():
    device = "cpu"

    model = UNet(
        in_channels=1,
        out_channels=1,
        spatial_dims=3,
        encoder_depth=[1, 1, 1, 1, 1],
        encoder_width=[32, 64, 128, 256, 512],
        strides=[1, 2, 2, 2, 2],
        decoder_depth=[1, 1, 1, 1],
        num_deep_supr=1,   # 🔴 强烈建议先设 1
      
    ).to(device)

    model.eval()

    # 假数据（典型 3D patch）
    x = torch.randn(1, 2, 64, 64, 64).to(device)

    with torch.no_grad():
        out = model(x)
    pred_shared = out["pred_shared"]
    pred_spec_m1 = out["pred_spec_m1"]
    pred_spec_m2 = out["pred_spec_m2"]    



    # ====== 检查 ======
    print("Deep supervision outputs:", len(pred_shared))
    print("Output shape:", pred_shared[0].shape)

    # print("Output shape:", y[0].shape, y[1].shape, y[2].shape)

    print("✅ UNet forward test passed!")


if __name__ == "__main__":
    from utils.helpers import as_tuple, prod, wrap_class
    test_unet_forward()
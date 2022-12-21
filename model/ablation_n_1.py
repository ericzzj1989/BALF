import torch
import torch.nn as nn
import einops

from model.decoder import DetectorHead


def block_images_einops(x, patch_size):
    """Image to patches."""
    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x

def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
        x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x

class GridGatingUnit(nn.Module):
    """A SpatialGatingUnit as defined in the gMLP paper.

    The 'spatial' dim is defined as the second last.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self, in_ch, intermediate_ch):
        super(GridGatingUnit, self).__init__()
        self.norm =  nn.LayerNorm(in_ch)
        self.dense = nn.Linear(intermediate_ch, intermediate_ch)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = v.permute(0, 3, 2, 1)
        v = self.dense(v)
        v = v.permute(0, 3, 2, 1)
        return u * (v + 1.)

class GridGmlpLayer(nn.Module):
    """Grid gMLP layer that performs global mixing of tokens."""
    def __init__(self, in_ch, grid_size, factor=2, dropout_rate=0.0):
        super(GridGmlpLayer, self).__init__()
        self.grid_size = grid_size
        intermediate_ch = grid_size[0] * grid_size[1]
        self.norm = nn.LayerNorm(in_ch)
        self.dense1 = nn.Linear(in_ch, in_ch * factor)
        self.act = nn.GELU()
        self.grid_gating_unit = GridGatingUnit(in_ch, intermediate_ch)
        self.dense2 = nn.Linear(in_ch, in_ch)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        n, h, w, num_channels = x.shape
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw
        x = block_images_einops(x, patch_size=(fh, fw))
        y = self.norm(x)
        y = self.dense1(y)
        y = self.act(y)
        y = self.grid_gating_unit(y)
        y = self.dense2(y)
        y = self.dropout(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x

class BlockGatingUnit(nn.Module):
    """A SpatialGatingUnit as defined in the gMLP paper.

    The 'spatial' dim is defined as the **second last**.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self, in_ch, intermediate_ch):
        super(BlockGatingUnit, self).__init__()
        self.norm =  nn.LayerNorm(in_ch)
        self.dense = nn.Linear(intermediate_ch, intermediate_ch)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = v.permute(0, 1, 3, 2)
        v = self.dense(v)
        v = v.permute(0, 1, 3, 2)
        return u * (v + 1.)

class BlockGmlpLayer(nn.Module):
    """Block gMLP layer that performs local mixing of tokens."""
    def __init__(self, in_ch, block_size, factor=2, dropout_rate=0.0):
        super(BlockGmlpLayer, self).__init__()
        self.block_size = block_size
        intermediate_ch = block_size[0] * block_size[1]
        self.norm = nn.LayerNorm(in_ch)
        self.dense1 = nn.Linear(in_ch, in_ch * factor)
        self.act = nn.GELU()
        self.block_gating_unit = BlockGatingUnit(in_ch, intermediate_ch)
        self.dense2 = nn.Linear(in_ch, in_ch)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        n, h, w, num_channels = x.shape
        fh, fw = self.block_size
        gh, gw = h // fh, w // fw
        x = block_images_einops(x, patch_size=(fh, fw))
        y = self.norm(x)
        y = self.dense1(y)
        y = self.act(y)
        y = self.block_gating_unit(y)
        y = self.dense2(y)
        y = self.dropout(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x

class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, in_ch, grid_size, block_size, grid_gmlp_factor=2,
                 block_gmlp_factor=2, input_proj_factor=2, dropout_rate=0.0):
        super(ResidualSplitHeadMultiAxisGmlpLayer, self).__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.dense1 = nn.Linear(in_ch, in_ch * input_proj_factor)
        self.act = nn.GELU()
        self.grid_gmlp_layer = GridGmlpLayer(in_ch, grid_size, grid_gmlp_factor, dropout_rate)
        self.block_gmlp_layer = BlockGmlpLayer(in_ch, block_size, block_gmlp_factor, dropout_rate)
        self.dense2 = nn.Linear(in_ch * input_proj_factor, in_ch)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        shortcut = x ## batch, h, w, num_channels
        x = self.norm(x)
        x = self.dense1(x)
        x = self.act(x)

        u, v = x.chunk(2, dim=-1)
        # GridGMLPLayer
        u = self.grid_gmlp_layer(u)

        # BlockGMLPLayer
        v = self.block_gmlp_layer(v)

        x = torch.cat([u, v], dim=-1)
        x = self.dense2(x)
        x = self.dropout(x)
        x = x + shortcut
        return x

class CALayer(nn.Module):
    """Squeeze-and-excitation block for channel attention.

    ref: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, in_ch, reduction_factor=4):
        super(CALayer, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction_factor),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction_factor, in_ch),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        n, num_channels, _, _ = x.shape
        squeeze_res = self.squeeze(x).view(n, num_channels)
        excite_res = self.excite(squeeze_res)
        f_scale = excite_res.view(n, num_channels, 1, 1)
        return x * f_scale

class ResidualChannelAttentionBlock(nn.Module):
    """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""
    def __init__(self, in_ch, reduction_factor=4, lrelu_slope=0.2):
        super(ResidualChannelAttentionBlock, self).__init__()
        self.norm = nn.LayerNorm(in_ch)
        # self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Linear(in_ch, in_ch)
        self.act = nn.LeakyReLU(negative_slope=lrelu_slope)
        # self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Linear(in_ch, in_ch)
        self.calayer = CALayer(in_ch, reduction_factor=reduction_factor)
    
    def forward(self, x):
        shortcut = x ## batch, h, w, num_channels
        x = self.norm(x)
        # x = x.permute(0, 3, 1, 2)
        # x = einops.rearrange(x, "b h w c -> b c h w")
        x = self.conv1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1)
        x = self.conv2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.calayer(x)
        x = x.permute(0, 2, 3, 1)
        # x = einops.rearrange(x, "b c h w -> b h w c")
        return x + shortcut

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, grid_size, block_size, grid_gmlp_factor=2,
                 block_gmlp_factor=2, input_proj_factor=2, channels_reduction=4, downsample=True):
        super(Down, self).__init__()
        self.downsample = downsample
        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Linear(in_ch, out_ch),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            # nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )
        self.residual_split_head_multi_axis_gmlp_layer = ResidualSplitHeadMultiAxisGmlpLayer(
            out_ch, grid_size, block_size, grid_gmlp_factor=grid_gmlp_factor,
            block_gmlp_factor=block_gmlp_factor, input_proj_factor=input_proj_factor, dropout_rate=0.0
        )
        self.residual_channel_attention_block = ResidualChannelAttentionBlock(out_ch, reduction_factor=channels_reduction)
        # self.conv_down = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv_down = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Linear(out_ch, out_ch)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        # x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.conv(x) # x.shape: b c h w
        shortcut_long = x
        # for i in range(1):
        x = self.residual_split_head_multi_axis_gmlp_layer(x)
        x = self.residual_channel_attention_block(x)
        # x = einops.rearrange(x, "b h w c -> b c h w")
        x = x + shortcut_long
        x = x.permute(0, 3, 1, 2)
        # x = einops.rearrange(x, "b h w c -> b c h w")
        if self.downsample:
            x_down = self.conv_down(x)
            return x_down
        else:
            x = x.permute(0, 2, 3, 1)
            # x = einops.rearrange(x, "b c h w -> b h w c")
            x = self.conv2(x)
            x = x.permute(0, 3, 1, 2)
            # x = einops.rearrange(x, "b h w c -> b c h w")
            return x
        
class Ablation_N_1(nn.Module):
    def __init__(self, model_cfg):
        super(Ablation_N_1, self).__init__()
        en_embed_dims = model_cfg['en_embed_dims']
        grid_size = model_cfg['grid_size']
        block_size = model_cfg['block_size']
        grid_gmlp_factor = model_cfg['grid_gmlp_factor']
        block_gmlp_factor = model_cfg['block_gmlp_factor']
        input_proj_factor = model_cfg['input_proj_factor']
        channels_reduction = model_cfg['channels_reduction']
        cell_size = model_cfg['cell_size']


        self.down1 = Down(en_embed_dims[0],en_embed_dims[1], grid_size=grid_size, block_size=block_size,
            grid_gmlp_factor=grid_gmlp_factor, block_gmlp_factor=block_gmlp_factor,
            input_proj_factor=input_proj_factor, channels_reduction=channels_reduction, downsample=True
        )
        # self.down2 = Down(en_embed_dims[1],en_embed_dims[2], grid_size=grid_size, block_size=block_size,
        #     grid_gmlp_factor=grid_gmlp_factor, block_gmlp_factor=block_gmlp_factor,
        #     input_proj_factor=input_proj_factor, channels_reduction=channels_reduction, downsample=True
        # )
        # self.down3 = Down(en_embed_dims[2],en_embed_dims[3], grid_size=grid_size, block_size=block_size,
        #     grid_gmlp_factor=grid_gmlp_factor, block_gmlp_factor=block_gmlp_factor,
        #     input_proj_factor=input_proj_factor, channels_reduction=channels_reduction, downsample=True
        # )
        # self.down4= Down(en_embed_dims[3],en_embed_dims[4], grid_size=grid_size, block_size=block_size,
        #     grid_gmlp_factor=grid_gmlp_factor, block_gmlp_factor=block_gmlp_factor,
        #     input_proj_factor=input_proj_factor, channels_reduction=channels_reduction, downsample=False
        # )

        self.detector_head = DetectorHead(input_channel=en_embed_dims[1], cell_size=cell_size)

    def forward(self, x):
        x = self.down1(x)
        # x = self.down2(x)
        # x = self.down3(x)
        # feat_map = self.down4(x)
        
        outputs = self.detector_head(x)
        return outputs
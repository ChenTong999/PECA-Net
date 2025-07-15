import torch.nn as nn
import torch
from PECA_Net.network_architecture.dynunet_block import UnetResBlock



class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.PEA_PCA = PEA_PCA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size,
                                   num_heads=num_heads, channel_attn_drop=dropout_rate,
                                   spatial_attn_drop=dropout_rate)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv52 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)  # x.shape [B, H * W * D, C]

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.PEA_PCA(self.norm(x))

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        attn = self.conv52(attn)
        x = attn_skip + self.conv8(attn)
        return x

###############################################################################
class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv1d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):

        q = self.Conv1x1(U)  # U:[bs,c,N] to q:[bs,1,N]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.Conv_Squeeze = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv1d(in_channels // 2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)  # shape: [bs, c, N] to [bs, c, 1]
        z = self.Conv_Squeeze(z)  # shape: [bs, c/2]
        z = self.Conv_Excitation(z)  # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)


class PEA_PCA(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """

    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False, channel_attn_drop=0.1,
                 spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.proj_size = proj_size

        self.M_k = nn.Parameter(torch.randn(hidden_size // num_heads, input_size, requires_grad=True) * 0.01)
        self.M_v_CA = nn.Parameter(torch.randn(hidden_size // num_heads, input_size, requires_grad=True) * 0.01)
        self.M_v_SA = nn.Parameter(torch.randn(hidden_size // num_heads, proj_size, requires_grad=True) * 0.01)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkvv = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        # E and F are projection matrices used in spatial attention module and channel attention to project keys and values from N-dimension or d-dimension to P-dimension
        self.E = nn.Linear(input_size, proj_size)

        self.attn_drop_1 = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 4))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 4))
        self.out_proj3 = nn.Linear(hidden_size, int(hidden_size // 4))
        self.out_proj4 = nn.Linear(hidden_size, int(hidden_size // 4))

        self.sSE = sSE(in_channels=int(hidden_size // 4))
        self.cSE = cSE(in_channels=int(hidden_size // 4))

    def forward(self, x):
        B, N, C = x.shape

        q = self.qkvv(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q_shared = q.permute(0, 2, 1, 3)  # [B, self.num_heads, N, C // self.num_heads]
        q_shared = q_shared.transpose(-2, -1)  # [B, self.num_heads, C // self.num_heads, N]

        M_k_projected = self.E(self.M_k)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        M_k = torch.nn.functional.normalize(self.M_k, dim=-1)

        attn_CA = (q_shared @ M_k.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-2)
        attn_CA = attn_CA / (1e-9 + attn_CA.sum(dim=-1, keepdim=True))
        attn_CA = self.attn_drop_1(attn_CA)

        x_CA = (attn_CA @ self.M_v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.transpose(-2, -1) @ M_k_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-2)
        attn_SA = attn_SA / (1e-9 + attn_SA.sum(dim=-1, keepdim=True))
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ self.M_v_SA.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA1 = self.out_proj(x_SA)
        x_CA1 = self.out_proj2(x_CA)

        x_SA2 = self.out_proj3(x)
        x_CA2 = self.out_proj4(x)
        x_SA2 = self.sSE(x_SA2.transpose(-2, -1)).transpose(-2, -1)
        x_CA2 = self.cSE(x_CA2.transpose(-2, -1)).transpose(-2, -1)

        x = torch.cat((x_SA1, x_SA2, x_CA1, x_CA2), dim=-1)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
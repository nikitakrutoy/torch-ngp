import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder


class SDFNetwork(nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 num_layers=3,
                 skips=[],
                 hidden_dim=64,
                 clip_sdf=None,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf

        self.encoder, self.in_dim = get_encoder(encoding)

        backbone = []

        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            elif l in self.skips:
                in_dim = self.hidden_dim + self.in_dim
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1
            else:
                out_dim = self.hidden_dim
            
            backbone.append(nn.Linear(in_dim, out_dim, bias=False))

        self.backbone = nn.ModuleList(backbone)

    
    def forward(self, x):
        # x: [B, 3]

        x = self.encoder(x)

        h = x
        for l in range(self.num_layers):
            if l in self.skips:
                h = torch.cat([h, x], dim=-1)
            h = self.backbone[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h
    
    def get_params(self):
        return [
            {'name': 'encoding', 'params': self.encoder.parameters()},
            {'name': 'net', 'params': self.backbone.parameters(), 'weight_decay': 1e-6},
        ]
    

class SqueezeSDFNetwork(nn.Module):

    def create_backbone(self, in_dim, out_dim, hidden_dim, num_layers):
        backbone = []
        for l in range(num_layers):
            if l == 0:
                in_dim_ = in_dim
            elif l in self.skips:
                in_dim_ = hidden_dim + in_dim
            else:
                in_dim_ = hidden_dim
 
            if l == num_layers - 1:
                out_dim_ = out_dim
            else:
                out_dim_ = hidden_dim
            
            backbone.append(nn.Linear(in_dim_, out_dim_, bias=False))


        return nn.ModuleList(backbone)

    def __init__(self,
                 encoding="hashgrid",
                 num_layers=3,
                 out_dim=8,
                 skips=[],
                 hidden_dim=128,
                 clip_sdf=None,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf

        self.out_dim = out_dim
        self.resolution = 2048
        self.encoder_xy, self.in_dim_xy = get_encoder(encoding, input_dim=2, desired_resolution=self.resolution, num_levels=23, level_dim=8)
        self.encoder_yz, self.in_dim_yz = get_encoder(encoding, input_dim=2, desired_resolution=self.resolution, num_levels=23, level_dim=8)
        self.encoder_zx, self.in_dim_zx = get_encoder(encoding, input_dim=2, desired_resolution=self.resolution, num_levels=23, level_dim=8)
        # self.encoder_xz, self.in_dim_xz = get_encoder(encoding, input_dim=2, desired_resolution=self.resolution, num_levels=23, level_dim=8)
        # self.encoder_zy, self.in_dim_zy = get_encoder(encoding, input_dim=2, desired_resolution=self.resolution, num_levels=23, level_dim=8)
        # self.encoder_yx, self.in_dim_yx = get_encoder(encoding, input_dim=2, desired_resolution=self.resolution, num_levels=23, level_dim=8)


        self.backbone_xy = self.create_backbone(self.in_dim_xy, self.out_dim, self.hidden_dim, self.num_layers)
        self.backbone_yz = self.create_backbone(self.in_dim_yz, self.out_dim, self.hidden_dim, self.num_layers)
        self.backbone_zx = self.create_backbone(self.in_dim_zx, self.out_dim, self.hidden_dim, self.num_layers)

        # self.backbone_xz = self.create_backbone(self.in_dim_xz, self.out_dim, self.hidden_dim, self.num_layers)
        # self.backbone_zy = self.create_backbone(self.in_dim_zy, self.out_dim, self.hidden_dim, self.num_layers)
        # self.backbone_yx = self.create_backbone(self.in_dim_yx, self.out_dim, self.hidden_dim, self.num_layers)




        self.head = torch.nn.Sequential(
            torch.nn.Linear(3, 1, bias=False),
        )

    def forward_backbone(self, x, backbone, encoder):
        x = encoder(x)
        h = x
        for l in range(self.num_layers):
            if l in self.skips:
                h = torch.cat([h, x], dim=-1)
            h = backbone[l](h)
            if l in self.skips:
                h = F.relu(h)
            if l != self.num_layers - 1:
                h = F.relu(h)
        return h
    
    def forward(self, x):
        # x: [B, 3]

        # x = self.encoder(x)

        xy = x[:, [0, 1]]
        yz = x[:, [1, 2]]
        zx = x[:, [2, 0]]
        # yx = x[:, [1, 0]]
        # zy = x[:, [2, 1]]
        # xz = x[:, [0, 2]]

        h_xy = self.forward_backbone(xy, self.backbone_xy, self.encoder_xy)
        h_yz = self.forward_backbone(yz, self.backbone_yz, self.encoder_yz)
        h_zx = self.forward_backbone(zx, self.backbone_zx, self.encoder_zx)
        # h_yx = self.forward_backbone(yx, self.backbone_yx, self.encoder_yx)
        # h_zy = self.forward_backbone(zy, self.backbone_zy, self.encoder_zy)
        # h_xz = self.forward_backbone(xz, self.backbone_xz, self.encoder_xz)

        h1 = (h_xy * h_yz).sum(1)[:, None]
        h2 = (h_yz * h_zx).sum(1)[:, None]
        h3 = (h_zx * h_xy).sum(1)[:, None]


        h = torch.cat([h1, h2, h3], dim=1)

        h = self.head(h)

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h
    
    def get_params(self):
        return [
            {'name': 'encoding', 'params': self.encoder_xy.parameters()},
            {'name': 'encoding', 'params': self.encoder_yz.parameters()},
            {'name': 'encoding', 'params': self.encoder_zx.parameters()},
            # {'name': 'encoding', 'params': self.encoder_yx.parameters()},
            # {'name': 'encoding', 'params': self.encoder_zy.parameters()},
            # {'name': 'encoding', 'params': self.encoder_xz.parameters()},

            {'name': 'net', 'params': self.backbone_xy.parameters(), 'weight_decay': 1e-6},
            {'name': 'net', 'params': self.backbone_yz.parameters(), 'weight_decay': 1e-6},
            {'name': 'net', 'params': self.backbone_zx.parameters(), 'weight_decay': 1e-6},
            # {'name': 'net', 'params': self.backbone_yx.parameters(), 'weight_decay': 1e-6},
            # {'name': 'net', 'params': self.backbone_zy.parameters(), 'weight_decay': 1e-6},
            # {'name': 'net', 'params': self.backbone_xz.parameters(), 'weight_decay': 1e-6},

            {'name': 'net', 'params': self.head.parameters(), 'weight_decay': 1e-6},
        ]
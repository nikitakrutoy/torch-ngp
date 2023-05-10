import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class ResidualBlock(nn.Module):
    def __init__(
        self,
        num_layers=4,
        in_features=64,
        hidden_dim=64, 
        bias=True
    ):
        super().__init__()

        self.num_layers = num_layers
        self.in_features = in_features
        self.skips = []
        net = []
        for l in range(num_layers):
            in_dim = in_features if l == 0 else hidden_dim
            out_dim = in_features if l == num_layers - 1 else hidden_dim
            net.append(nn.Linear(in_dim, out_dim, bias=bias))
            net.append(nn.InstanceNorm1d(out_dim))
            # net.append(nn.BatchNorm1d(num_features=out_dim))
            if l != num_layers - 1:
                net.append(nn.LeakyReLU())
                # net.append(nn.Sigmoid())
            # if l != num_layers - 1:
            #     net.append(nn.Dropout(p=0.9))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        h = x
        for layer in self.net:
            h = layer(h)

            # if l != self.num_layers - 1:
            #     h = F.relu(h, inplace=True) 

        return F.leaky_relu(h + x)

class NeTFMLP3(NeRFRenderer):

    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 resblock_in_dim=512,
                 resblock_num_layers=4,
                 resblock_hidden_dim=512,
                 resblock_num=2,
                 num_layers_bg=2,
                 hidden_dim_bg=128,
                 bound=1,
                 **kwargs,
                ):
        super().__init__(bound, **kwargs)
        
        # sigma network
        self.resblock_num = resblock_num
        self.resblock_in_dim = resblock_in_dim
        self.resblock_hidden_dim = resblock_hidden_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound, num_levels=23, level_dim=4, base_resolution=8, log2_hashmap_size=21)
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir, degree=6)

        sigma_net = []
        for l in range(resblock_num + 2):
            if l == 0:
                in_dim = self.in_dim + self.in_dim_dir
                out_dim = self.resblock_in_dim
                sigma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))
                sigma_net.append(nn.LeakyReLU())
            elif l == resblock_num + 1:
                in_dim = self.resblock_in_dim
                out_dim = 1
                sigma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))

            else:
                sigma_net.append(ResidualBlock(resblock_num_layers, resblock_in_dim, resblock_hidden_dim))
                # sigma_net.append(nn.BatchNorm1d(num_features=resblock_in_dim))


        self.sigma_net = nn.ModuleList(sigma_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

                # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)
        d = self.encoder_dir(d)
        h_ = torch.cat([x, d], dim=-1)
        h = h_.clone()

        for l in range(len(self.sigma_net)):
            h = self.sigma_net[l](h)
            if torch.any(torch.isnan(h)):
                print("Got nans")
                # exit(1)
            # if l != self.resblock_num + 1:
            #     h = F.leaky_relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        # sigma = trunc_exp(h[..., 0])
        sigma = h[..., 0]
        # geo_feat = h[..., 1:]
        sigma_ = sigma.unsqueeze(-1)
        color = torch.hstack([sigma_, sigma_, sigma_])

        return sigma, color

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    def density(self, x, d):
        # x: [N, 3], in [-bound, bound]
        result = self.forward(x, d)
        return {
            'sigma': result[0],
            'color': result[1]
        }


    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params


class NeTFMLP4(NeRFRenderer):

    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 resblock_in_dim=512,
                 resblock_num_layers=4,
                 resblock_hidden_dim=512,
                 resblock_num=2,
                 num_layers_bg=2,
                 hidden_dim_bg=128,
                 bound=1,
                 **kwargs,
                ):
        super().__init__(bound, **kwargs)
        
        # sigma network
        self.resblock_num = resblock_num
        self.resblock_in_dim = resblock_in_dim
        self.resblock_hidden_dim = resblock_hidden_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir, degree=4)

        self.interpolate = False
        self.CP = 8
        self.CD = 4

        sigma_net = []
        for l in range(resblock_num + 2):
            if l == 0:
                in_dim = self.in_dim
                out_dim = self.resblock_in_dim
                sigma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))
                sigma_net.append(nn.LeakyReLU())
            elif l == resblock_num + 1:
                in_dim = self.resblock_in_dim
                out_dim = 8
                sigma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))

            else:
                sigma_net.append(ResidualBlock(resblock_num_layers, resblock_in_dim, resblock_hidden_dim))
                # sigma_net.append(nn.BatchNorm1d(num_features=resblock_in_dim))

        self.sigma_net = nn.ModuleList(sigma_net)
        

        gamma_net = []
        for l in range(resblock_num + 2):
            if l == 0:
                in_dim = self.in_dim_dir
                out_dim = self.resblock_in_dim
                gamma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))
                gamma_net.append(nn.LeakyReLU())
            elif l == resblock_num + 1:
                in_dim = self.resblock_in_dim
                out_dim = 8
                gamma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))

            else:
                gamma_net.append(ResidualBlock(resblock_num_layers, resblock_in_dim, resblock_hidden_dim))
                # sigma_net.append(nn.BatchNorm1d(num_features=resblock_in_dim))


        self.gamma_net = nn.ModuleList(gamma_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    def interpolate_pos(self, x):
        N = 2048
        L = (4 * 0.33) * 2
        K = L / N
        coords = []
        for foo in [torch.ceil, torch.floor]:
            for bar in [torch.ceil, torch.floor]:
                for mas in [torch.ceil, torch.floor]:
                    coords.append(
                        torch.stack([
                            foo(x[:, 0] / K) * K, 
                            bar(x[:, 1] / K) * K, 
                            mas(x[:, 2] / K) * K
                        ]).transpose(0, 1)
                    )
        C = self.CP
        X = torch.zeros(x.shape[0] * C, x.shape[1]).to("cuda")
        for i in range(C):
            X[i::C] = coords[i]
        diff = X.view(x.shape[0], C, x.shape[1]) - x.view(x.shape[0], 1, x.shape[1]).expand((x.shape[0], C, x.shape[1]))
        area = diff.abs().prod(dim=2) 
        weights = area / (K ** 3)
        return X, weights
    
    def interpolate_dir(self, d):
        z = d[:, 2]
        y = d[:, 1]
        x = d[:, 0]
        theta = torch.acos(z) % (2 * torch.pi)
        phi = torch.atan2(y, x) % (2 * torch.pi)
        d2 = torch.vstack([theta, phi]).transpose(0, 1)
        N = 2048
        L = 2 * torch.pi
        K = L / N
        coords = []
        for foo in [torch.ceil, torch.floor]:
            for bar in [torch.ceil, torch.floor]:
                    coords.append(
                        torch.stack([
                            foo(d2[:, 0] / K) * K, 
                            bar(d2[:, 1] / K) * K 
                        ]).transpose(0, 1)
                    )
        C = self.CD
        D = torch.zeros(d.shape[0] * C, 2).to("cuda")
        for i in range(C):
            D[i::C] = coords[i]
        diff = D.view(d2.shape[0], C, d2.shape[1]) - d2.view(d2.shape[0], 1, d2.shape[1]).expand((d2.shape[0], C, d2.shape[1]))
        area = diff.abs().prod(dim=2) 
        weights = area / (K ** 2)

        theta = D[:, 0]
        phi = D[:, 1]
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        D = torch.stack([x, y, z]).transpose(0, 1)
        return D, weights

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

                # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        if self.interpolate:
            x, weights_pos = self.interpolate_pos(x)
            d, weights_dir = self.interpolate_dir(d)

        # sigma
        x = self.encoder(x, bound=self.bound)
        d = self.encoder_dir(d)

        h = x.clone()
        for l in range(len(self.sigma_net)):
            h = self.sigma_net[l](h)
            if torch.any(torch.isnan(h)):
                print("Got nans")
                # exit(1)
            # if l != self.resblock_num + 1:
            #     h = F.leaky_relu(h, inplace=True)
        # sigma = h[..., 0]
        sigma = h.clone()
        

        h = d.clone()
        for l in range(len(self.gamma_net)):
            h = self.gamma_net[l](h)
            if torch.any(torch.isnan(h)):
                print("Got nans")
                # exit(1)
            # if l != self.resblock_num + 1:
            #     h = F.leaky_relu(h, inplace=True)
        # gamma = h[..., 0]
        gamma = h.clone()

        if self.interpolate:
            weights_pos = weights_pos\
                .view(sigma.shape[0] // self.CP, self.CP, 1)\
                .expand((sigma.shape[0] // self.CP, self.CP, sigma.shape[1]))
            sigma = sigma.view(sigma.shape[0] // self.CP, self.CP, sigma.shape[1]) 
            sigma = (sigma * weights_pos).sum(dim=1)

            weights_dir = weights_dir\
                .view(gamma.shape[0] // self.CD, self.CD, 1)\
                .expand((gamma.shape[0] // self.CD, self.CD, gamma.shape[1]))
            gamma = gamma.view(gamma.shape[0] // self.CD, self.CD, gamma.shape[1])
            gamma = (gamma * weights_dir).sum(dim=1)


        result = (sigma * gamma).sum(-1)
        

        #sigma = F.relu(h[..., 0])
        # sigma = trunc_exp(h[..., 0])
        # geo_feat = h[..., 1:]
        result_ = result.unsqueeze(-1)
        color = torch.hstack([result_, result_, result_])

        return result, color

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    def density(self, x, d):
        # x: [N, 3], in [-bound, bound]
        result = self.forward(x, d)
        return {
            'sigma': result[0],
            'color': result[1]
        }


    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.gamma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
    
class KiloNeTF(NeRFRenderer):

    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 resblock_in_dim=512,
                 resblock_num_layers=4,
                 resblock_hidden_dim=512,
                 resblock_num=2,
                 num_layers_bg=2,
                 hidden_dim_bg=128,
                 resolution=2,
                 bound=1,
                 shared_encoders=False,
                 **kwargs,
                ):
        super().__init__(bound, **kwargs)
        
        # sigma network
        self.resblock_num = resblock_num
        self.resblock_in_dim = resblock_in_dim
        self.resblock_hidden_dim = resblock_hidden_dim
        self.encoding = encoding
        self.shader_encoders = shared_encoders
        if shared_encoders:
            self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound, degree=4)
            self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir, degree=4)
        else:
            self.encoders = []
            self.encoders_dir = []
            for _ in range(resolution ** 3):
                encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound / resolution, degree=4)
                encoder_dir, self.in_dim_dir = get_encoder(encoding_dir, degree=4)
                self.encoders_dir.append(encoder_dir)
                self.encoders.append(encoder)
            self.encoders = nn.ModuleList(self.encoders)
            self.encoder_dir = nn.ModuleList(self.encoders_dir)
        self.bound = bound
        self.resolution = resolution

        self.interpolate = False
        self.CP = 8
        self.CD = 4
        self.sigma_nets = []
        for _ in range(resolution ** 3):
            sigma_net = []
            for l in range(resblock_num + 2):
                if l == 0:
                    in_dim = self.in_dim
                    out_dim = self.resblock_in_dim
                    sigma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                    # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))
                    sigma_net.append(nn.LeakyReLU())
                elif l == resblock_num + 1:
                    in_dim = self.resblock_in_dim
                    out_dim = 8
                    sigma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                    # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))

                else:
                    sigma_net.append(ResidualBlock(resblock_num_layers, resblock_in_dim, resblock_hidden_dim))
                    # sigma_net.append(nn.BatchNorm1d(num_features=resblock_in_dim))

            self.sigma_nets.append(nn.ModuleList(sigma_net))
        self.sigma_nets = nn.ModuleList(self.sigma_nets)
        
        self.gamma_nets = []
        for _ in range(resolution ** 3):
            gamma_net = []
            for l in range(resblock_num + 2):
                if l == 0:
                    in_dim = self.in_dim_dir
                    out_dim = self.resblock_in_dim
                    gamma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                    # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))
                    gamma_net.append(nn.LeakyReLU())
                elif l == resblock_num + 1:
                    in_dim = self.resblock_in_dim
                    out_dim = 8
                    gamma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                    # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))

                else:
                    gamma_net.append(ResidualBlock(resblock_num_layers, resblock_in_dim, resblock_hidden_dim))
                    # sigma_net.append(nn.BatchNorm1d(num_features=resblock_in_dim))
            gamma_net = nn.ModuleList(gamma_net)
            self.gamma_nets.append(gamma_net)
        self.gamma_nets = nn.ModuleList(self.gamma_nets)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    def interpolate_pos(self, x):
        N = 2048
        L = (4 * 0.33) * 2
        K = L / N
        coords = []
        for foo in [torch.ceil, torch.floor]:
            for bar in [torch.ceil, torch.floor]:
                for mas in [torch.ceil, torch.floor]:
                    coords.append(
                        torch.stack([
                            foo(x[:, 0] / K) * K, 
                            bar(x[:, 1] / K) * K, 
                            mas(x[:, 2] / K) * K
                        ]).transpose(0, 1)
                    )
        C = self.CP
        X = torch.zeros(x.shape[0] * C, x.shape[1]).to("cuda")
        for i in range(C):
            X[i::C] = coords[i]
        diff = X.view(x.shape[0], C, x.shape[1]) - x.view(x.shape[0], 1, x.shape[1]).expand((x.shape[0], C, x.shape[1]))
        area = diff.abs().prod(dim=2) 
        weights = area / (K ** 3)
        return X, weights
    
    def interpolate_dir(self, d):
        z = d[:, 2]
        y = d[:, 1]
        x = d[:, 0]
        theta = torch.acos(z) % (2 * torch.pi)
        phi = torch.atan2(y, x) % (2 * torch.pi)
        d2 = torch.vstack([theta, phi]).transpose(0, 1)
        N = 2048
        L = 2 * torch.pi
        K = L / N
        coords = []
        for foo in [torch.ceil, torch.floor]:
            for bar in [torch.ceil, torch.floor]:
                    coords.append(
                        torch.stack([
                            foo(d2[:, 0] / K) * K, 
                            bar(d2[:, 1] / K) * K 
                        ]).transpose(0, 1)
                    )
        C = self.CD
        D = torch.zeros(d.shape[0] * C, 2).to("cuda")
        for i in range(C):
            D[i::C] = coords[i]
        diff = D.view(d2.shape[0], C, d2.shape[1]) - d2.view(d2.shape[0], 1, d2.shape[1]).expand((d2.shape[0], C, d2.shape[1]))
        area = diff.abs().prod(dim=2) 
        weights = area / (K ** 2)

        theta = D[:, 0]
        phi = D[:, 1]
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        D = torch.stack([x, y, z]).transpose(0, 1)
        return D, weights
    
    def forward(self, x, d):
        x_ = torch.floor(x / (2 * self.bound / self.resolution)) + self.resolution // 2
        x_ = torch.clamp(x_, 0, self.resolution - 1)
        x_ = x_[:, 0] * self.resolution * self.resolution + x_[:, 1] * self.resolution + x_[:, 2]
        c = (x_ - self.resolution // 2 + 0.5) * (2 * self.bound / self.resolution)
        bids = x_.int()
        result = torch.zeros((x.shape[0])).to("cuda")
        color = torch.zeros((x.shape[0], 3)).to("cuda")
        for bid in torch.unique(bids):
        # for bid in [30]:
            # print(bid)
            pos = x[x_ == bid]
            dir = d[x_ == bid]
            centers = c[x_ == bid]
            result_, color_ = self.forward_(pos, dir, centers, int(bid))
            result[x_ == bid] = result_
            color[x_ == bid] = color_

        return result, color, bids
    
    def forward_(self, x, d, c, i):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

                # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        if self.interpolate:
            x, weights_pos = self.interpolate_pos(x)
            d, weights_dir = self.interpolate_dir(d)

        # sigma
        if self.shader_encoders:
            x = self.encoder(x, bound=self.bound)
            d = self.encoder_dir(d)
        else:
            if self.encoding == "hashgrid":
                x = self.encoders[i](x - c, bound=self.bound / self.resolution)
            else:
                x = self.encoders[i](x, bound=self.bound / self.resolution)
            d = self.encoder_dir[i](d)


        h = x.clone()
        for l in range(len(self.sigma_nets[i])):
            h = self.sigma_nets[i][l](h)
            if torch.any(torch.isnan(h)):
                print("Got nans")
                # exit(1)
            # if l != self.resblock_num + 1:
            #     h = F.leaky_relu(h, inplace=True)
        # sigma = h[..., 0]
        sigma = h.clone()
        

        h = d.clone()
        for l in range(len(self.gamma_nets[i])):
            h = self.gamma_nets[i][l](h)
            if torch.any(torch.isnan(h)):
                print("Got nans")
                # exit(1)
            # if l != self.resblock_num + 1:
            #     h = F.leaky_relu(h, inplace=True)
        # gamma = h[..., 0]
        gamma = h.clone()

        if self.interpolate:
            weights_pos = weights_pos\
                .view(sigma.shape[0] // self.CP, self.CP, 1)\
                .expand((sigma.shape[0] // self.CP, self.CP, sigma.shape[1]))
            sigma = sigma.view(sigma.shape[0] // self.CP, self.CP, sigma.shape[1]) 
            sigma = (sigma * weights_pos).sum(dim=1)

            weights_dir = weights_dir\
                .view(gamma.shape[0] // self.CD, self.CD, 1)\
                .expand((gamma.shape[0] // self.CD, self.CD, gamma.shape[1]))
            gamma = gamma.view(gamma.shape[0] // self.CD, self.CD, gamma.shape[1])
            gamma = (gamma * weights_dir).sum(dim=1)


        result = (sigma * gamma).sum(-1)
        

        #sigma = F.relu(h[..., 0])
        # sigma = trunc_exp(h[..., 0])
        # geo_feat = h[..., 1:]
        result_ = result.unsqueeze(-1)
        color = torch.hstack([result_, result_, result_])

        return result, color

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    def density(self, x, d):
        # x: [N, 3], in [-bound, bound]
        result = self.forward(x, d)
        return {
            'sigma': result[0],
            'color': result[1],
            'bids' : result[2]
        }


    def get_params(self, lr):
        if self.shader_encoders:
            encoder_params = [{'params': self.encoder.parameters(), 'lr': lr}]
            encoder_dir_params = [{'params': self.encoder_dir.parameters(), 'lr': lr}]
        else:
            encoder_params = [{'params': net.parameters(), 'lr': lr} for net in self.encoders]
            encoder_dir_params = [{'params': net.parameters(), 'lr': lr} for net in self.encoders_dir]
        params = encoder_params + encoder_dir_params +\
            [{'params': net.parameters(), 'lr': lr} for net in self.sigma_nets] +\
            [{'params': net.parameters(), 'lr': lr} for net in self.gamma_nets]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params



class NeTFMLP5(NeRFRenderer):

    def create_sigma_network(self, in_dim):
        sigma_net = []
        for l in range(self.resblock_num + 2):
            if l == 0:
                in_dim = in_dim
                out_dim = self.resblock_in_dim
                sigma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))
                sigma_net.append(nn.LeakyReLU())
            elif l == self.resblock_num + 1:
                in_dim = self.resblock_in_dim
                out_dim = self.last_dim
                sigma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))

            else:
                sigma_net.append(ResidualBlock(self.resblock_num_layers, self.resblock_in_dim, self.resblock_hidden_dim))
                # sigma_net.append(nn.BatchNorm1d(num_features=resblock_in_dim))

        return nn.ModuleList(sigma_net)

    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 resblock_in_dim=512,
                 resblock_num_layers=2,
                 resblock_hidden_dim=512,
                 resblock_num=2,
                 num_layers_bg=2,
                 hidden_dim_bg=128,
                 bound=1,
                 **kwargs,
                ):
        super().__init__(bound, **kwargs)
        
        # sigma network
        self.resblock_num = resblock_num
        self.resblock_num_layers = resblock_num_layers
        self.resblock_in_dim = resblock_in_dim
        self.resblock_hidden_dim = resblock_hidden_dim
        self.encoder_xy, self.in_dim_xy = get_encoder(encoding, input_dim=2, desired_resolution=2048 * bound, num_levels=23, level_dim=4)
        self.encoder_yz, self.in_dim_yz = get_encoder(encoding, input_dim=2, desired_resolution=2048 * bound, num_levels=23, level_dim=4)
        self.encoder_zx, self.in_dim_zx = get_encoder(encoding, input_dim=2, desired_resolution=2048 * bound, num_levels=23, level_dim=4)

        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir, degree=6)

        self.last_dim = 8

        self.sigma_net_xy = self.create_sigma_network(self.in_dim_xy)
        self.sigma_net_yz = self.create_sigma_network(self.in_dim_yz)
        self.sigma_net_zx = self.create_sigma_network(self.in_dim_zx)

        self.fusion = nn.Linear(3 * self.last_dim, self.last_dim)

        gamma_net = []
        for l in range(resblock_num + 2):
            if l == 0:
                in_dim = self.in_dim_dir
                out_dim = self.resblock_in_dim
                gamma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))
                gamma_net.append(nn.LeakyReLU())
            elif l == resblock_num + 1:
                in_dim = self.resblock_in_dim
                out_dim = self.last_dim
                gamma_net.append(nn.Linear(in_dim, out_dim, bias=True))
                # sigma_net.append(nn.BatchNorm1d(num_features=out_dim))

            else:
                gamma_net.append(ResidualBlock(resblock_num_layers, resblock_in_dim, resblock_hidden_dim))
                # sigma_net.append(nn.BatchNorm1d(num_features=resblock_in_dim))


        self.gamma_net = nn.ModuleList(gamma_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

                # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        xy = x[:, [0, 1]]
        yz = x[:, [1, 2]]
        zx = x[:, [0, 2]]
        xy = self.encoder_xy(xy, bound=self.bound)
        yz = self.encoder_yz(yz, bound=self.bound)
        zx = self.encoder_zx(zx, bound=self.bound)
        d = self.encoder_dir(d)

        h = xy.clone()
        for l in range(len(self.sigma_net_xy)):
            h = self.sigma_net_xy[l](h)
            if torch.any(torch.isnan(h)):
                print("Got nans")
        sigma_xy = h.clone()

        h = yz.clone()
        for l in range(len(self.sigma_net_yz)):
            h = self.sigma_net_yz[l](h)
            if torch.any(torch.isnan(h)):
                print("Got nans")
        sigma_yz = h.clone()

        h = zx.clone()
        for l in range(len(self.sigma_net_zx)):
            h = self.sigma_net_zx[l](h)
            if torch.any(torch.isnan(h)):
                print("Got nans")
        sigma_zx = h.clone()

        h = d.clone()
        for l in range(len(self.gamma_net)):
            h = self.gamma_net[l](h)
            if torch.any(torch.isnan(h)):
                print("Got nans")
        gamma = h.clone()

        sigma = self.fusion(torch.concat([sigma_xy, sigma_yz, sigma_zx], dim=1))
        # sigma = sigma_xy + sigma_yz + sigma_zx

        result = (sigma * gamma).sum(-1)
        
        result_ = result.unsqueeze(-1)
        color = torch.hstack([result_, result_, result_])

        return result, color

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    def density(self, x, d):
        # x: [N, 3], in [-bound, bound]
        result = self.forward(x, d)
        return {
            'sigma': result[0],
            'color': result[1]
        }


    def get_params(self, lr):

        params = [
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.sigma_net_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.sigma_net_yz.parameters(), 'lr': lr},
            {'params': self.encoder_zx.parameters(), 'lr': lr},
            {'params': self.sigma_net_zx.parameters(), 'lr': lr},
            {'params': self.gamma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.fusion.parameters(), 'lr': lr}
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
class NeTFMLP2(NeRFRenderer):

    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=8,
                 skips = [2, 4, 6],
                 hidden_dim=128,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 **kwargs,
                ):
        super().__init__(bound, **kwargs)
        
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution= 2048 * bound, num_levels=16, level_dim=4)
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim + self.in_dim_dir
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

                # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)
        d = self.encoder_dir(d)
        h_ = torch.cat([x, d], dim=-1)
        h = h_.clone()

        for l in range(self.num_layers):
            if l not in self.skips:
                h = self.sigma_net[l](h)
            else:
                h = self.sigma_net[l]( torch.cat([h, h_], dim=-1))

            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        # geo_feat = h[..., 1:]
        sigma_ = sigma.unsqueeze(-1)
        color = torch.hstack([sigma_, sigma_, sigma_])

        return sigma, color

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    def density(self, x, d):
        # x: [N, 3], in [-bound, bound]
        result = self.forward(x, d)
        return {
            'sigma': result[0],
            'color': result[1]
        }


    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None


    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params

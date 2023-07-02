import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import RegularGridInterpolator

import trimesh
import pysdf

class SampleBox:

    def __init__(self, path, num_samples):
        self.path = path
        self.num_samples = num_samples
        box = trimesh.load(path, force="mesh")
        vs = box.vertices
        vmin = vs.min(0)
        vmax = vs.max(0)
        self.center = (vmin + vmax) / 2
        self.scale = vmax - vmin

def map_color(value, cmap_name='viridis', vmin=None, vmax=None):
    # value: [N], float
    # return: RGB, [N, 3], float in [0, 1]
    import matplotlib.cm as cm
    if vmin is None: vmin = value.min()
    if vmax is None: vmax = value.max()
    value = (value - vmin) / (vmax - vmin) # range in [0, 1]
    cmap = cm.get_cmap(cmap_name) 
    rgb = cmap(value)[:, :3]  # will return rgba, we take only first 3 so we get rgb
    return rgb

def plot_pointcloud(pc, sdfs):
    # pc: [N, 3]
    # sdfs: [N, 1]
    color = map_color(sdfs.squeeze(1))
    pc = trimesh.PointCloud(pc, color)
    trimesh.Scene([pc]).show()    

def flat_meshgrid(grid_size):
    x_ = np.linspace(-1, 1., grid_size)
    y_ = np.linspace(-1, 1., grid_size)
    z_ = np.linspace(-1, 1., grid_size)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    return np.stack((x, y, z), axis=3).reshape((-1, 3))

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    # if packaging.version.parse(torch.__version__) < packaging.version.parse('1.10'):
    #     return torch.meshgrid(*args)
    # else:
    return torch.meshgrid(*args, indexing='ij')

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
                    val = torch.from_numpy(query_func(pts.numpy())).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [N, 1] --> [x, y, z]
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u
# SDF dataset
class SDFDataset(Dataset):
    def __init__(self, path, size=100, grid_size=16, num_samples=2**18, clip_sdf=None, sample_boxes=None):
        super().__init__()
        self.path = path
        self.sample_boxes = sample_boxes
        # load obj 
        self.mesh = trimesh.load(path, force='mesh')
        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        vs = self.mesh.vertices
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        self.v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * self.v_scale
        self.mesh.vertices = vs

        print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        if not self.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
        #trimesh.Scene([self.mesh]).show()

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size
        self.points_uniform = None
        self.surface_size = (self.num_samples * 4) // 8
        self.perturb_size = (self.surface_size * 4) // 8
        self.uniform_size = self.num_samples  - self.surface_size
        self.blob_size = 2 ** 6

        self.grid_size = grid_size
        self.grid = flat_meshgrid(self.grid_size)
        self.probs = None
        
        bounds_min = torch.FloatTensor([-1, -1, -1])
        bounds_max = torch.FloatTensor([1, 1, 1])
        foo = lambda x: -self.sdf_fn(x)
        N = self.grid_size
        self.U = extract_fields(bounds_min, bounds_max, resolution=N, query_func=foo)
        gridMin = -1
        gridMax = 1
        x = np.linspace(gridMin, gridMax, N)
        y = np.linspace(gridMin, gridMax, N)
        z = np.linspace(gridMin, gridMax, N)
        self.interpU = RegularGridInterpolator((x, y, z), self.U)

    def update_default(self, loss):
        self.points_uniform=None

    def update(self, loss):
        uniform_std = 1.0 / self.uniform_size ** 0.33
        loss_uniform = loss[:self.points_uniform.shape[0]]
        prob = torch.softmax(loss_uniform, dim=0).detach().cpu().numpy().flatten()
        idx = np.random.choice(
            np.arange(self.uniform_size, dtype=int), 
            size=int(self.uniform_size / self.blob_size),
            replace=False,
            p=prob
        )
        self.points_uniform = self.points_uniform[idx]
        self.points_uniform = np.repeat(self.points_uniform, self.blob_size, axis=0)
        self.points_uniform += (np.random.rand(self.uniform_size, 3) * 2 - 1) * uniform_std

    def update_probs(self, probs, only_surface):
        self.surface_size = self.num_samples  - 2 ** 4 if only_surface else (self.num_samples * 4) // 8

        self.perturb_size = (self.surface_size * 4) // 8
        self.uniform_size = self.num_samples  - self.surface_size
        self.probs = probs
    
    def __len__(self):
        return self.size

    def __getitem__(self, _):

        # online sampling
        sdfs = np.zeros((self.num_samples, 1))

        perturb_offsets = [0.1, 0.075, 0.05, 0.025, 0.01]
        perturb_offsets = np.repeat(perturb_offsets, self.perturb_size // len(perturb_offsets))
        self.perturb_size = len(perturb_offsets)
        # surface_size = self.num_samples * 7 // 8
        # perturb_size = self.num_samples * 3 // 8
        # uniform_size = self.num_samples  - surface_size
        # surface
        points_surface = self.mesh.sample(self.surface_size)
        # perturb surface
        points_surface[-self.perturb_size:] += perturb_offsets[:, None] * np.random.randn(self.perturb_size, 3)
        # random
        if self.probs is not None:
            idx = np.random.choice(
                np.arange(self.probs.size, dtype=int), 
                size=self.uniform_size,
                replace=False,
                p=self.probs
            )
            self.points_uniform = self.grid[idx] + (np.random.rand(self.uniform_size, 3) * 2 - 1) * (1 / self.grid_size)
        if self.points_uniform is None:
            self.points_uniform = np.random.rand(self.uniform_size, 3) * 2 - 1

        points = np.concatenate([points_surface, self.points_uniform], axis=0).astype(np.float32)

        sdfs[-(self.uniform_size + self.perturb_size):] = -self.sdf_fn(points[-(self.uniform_size + self.perturb_size):])[:,None].astype(np.float32)

        sdfs_interp = self.interpU(np.clip(points, -1, 1)).reshape(-1, 1)
        sdfs = sdfs - sdfs_interp

        if self.sample_boxes is not None and len(self.sample_boxes) > 0:
            box_points = []
            for box in self.sample_boxes:
                box_points.append(self.v_scale * (box.scale * np.random.randn(box.num_samples, 3) + box.center))
            box_points = np.concatenate(box_points).astype(np.float32)
            box_sdf = -self.sdf_fn(box_points)[:,None].astype(np.float32)

            points = np.concatenate([points, box_points])
            sdfs = np.concatenate([sdfs, box_sdf])
 
        # clip sdf
        if self.clip_sdf is not None:
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        results = {
            'sdfs': sdfs,
            'sdfs_interp': sdfs_interp,
            'points': points,
        }

        #plot_pointcloud(points, sdfs)

        return results
    
class UpdateDataset(Dataset):
    def __init__(self, path, grid_size=100):
        super().__init__()
        self.grid_size = grid_size
        self.size = grid_size ** 3
        self.mesh = trimesh.load(path, force='mesh')
        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        vs = self.mesh.vertices
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        self.v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * self.v_scale
        self.mesh.vertices = vs
        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)


    
    def __len__(self):
        return self.size

    def __getitem__(self, _):

        points = flat_meshgrid(self.grid_size).astype(np.float32)

        sdfs = -self.sdf_fn(points)[:,None].astype(np.float32)

        results = {
            'sdfs': sdfs,
            'points': points,
        }

        return results

class SDFLoader(DataLoader):
    counter = 1
    def update(self, loss):
        self.dataset.update_default(loss)

    def update_probs(self, probs):

        self.dataset.update_probs(probs, False)
        self.counter += 1
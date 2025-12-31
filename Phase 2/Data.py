import os 
import json 
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class LoadSyntheticDataset(Dataset): 
    def __init__(self, path_to_images, path_to_labels): 
        
        if not os.path.exists(path_to_images):
            raise FileNotFoundError(f"Images directory not found: {path_to_images}")
        
        if not os.path.exists(path_to_labels):
            raise FileNotFoundError(f"Labels file not found: {path_to_labels}")
            
        self.path_to_images = path_to_images
        all_files = os.listdir(path_to_images)
        self.images = [im for im in all_files if im.endswith('.png')]
        
        self.transform = transforms.ToTensor()
        
        try:
            with open(path_to_labels, 'r') as f: 
                self.labels = json.load(f)
            self.camera_angle_x = self.labels.get('camera_angle_x', None)
        except Exception as e:
            raise

    def __getitem__ (self, idx): 
        try:
            label = self.labels['frames'][idx]
            file_name = os.path.basename(label['file_path']) + '.png'
            img_path = os.path.join(self.path_to_images, file_name)
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
                
            image = Image.open(img_path).convert("RGB") 

            if self.transform: 
                image = self.transform(image)

            N_rays = 4096
            H, W = image.shape[1], image.shape[2]
            
            if self.camera_angle_x is not None:
                focal = W / (2 * np.tan(self.camera_angle_x / 2))
            else:
                focal = W / 2 

            i = torch.randint(0, W, (N_rays,))
            j = torch.randint(0, H, (N_rays,))
            
            rgb_gt = image[:, j, i].permute(1, 0)  # [N_rays, 3]

            x = (i.float() - W * 0.5) / focal
            y = (j.float() - H * 0.5) / focal
            z = -torch.ones_like(x)
            dirs = torch.stack([x, y, z], dim=-1)  # [N_rays, 3]

            c2w = torch.tensor(label['transform_matrix'], dtype=torch.float32)  # [4, 4]
            rays_d = (dirs @ c2w[:3, :3].T).float()  # Rotate ray directions
            rays_o = c2w[:3, 3].expand(rays_d.shape)  # [N_rays, 3]

            near, far = 2.0, 6.0
            t_vals = torch.linspace(0., 1., steps=64)
            z_vals = near * (1. - t_vals) + far * t_vals  # [64]
            z_vals = z_vals.expand(N_rays, -1)  # [N_rays, 64]

            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand

            points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # [N_rays, 64, 3]

            return {
                'points': points,
                'rays_d': rays_d,
                'rgb_gt': rgb_gt,
                'z_vals': z_vals
            }
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def __len__(self): 
        return len(self.images)

class CustomDataloader: 
    def __init__(self, batch_size, path_to_images=None, path_to_labels=None): 
        
        if path_to_images is None or path_to_labels is None:
            raise ValueError("Both path_to_images and path_to_labels must be provided")
            
        self.dataset = LoadSyntheticDataset(
                path_to_images=path_to_images, 
                path_to_labels=path_to_labels  
            )
        self.batch_size = batch_size
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def __iter__(self):
        return iter(self.loader)
        
    def __len__(self):
        return len(self.loader)

workspace_root = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(workspace_root, 'nerf_synthetic', 'chair')
data_path = os.path.join(base_path, 'train')
transforms_path = os.path.join(base_path, 'transforms_train.json')

batch_size = 1
        
dataloader = CustomDataloader(batch_size, data_path, transforms_path)
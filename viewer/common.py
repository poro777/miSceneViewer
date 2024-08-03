import os
import numpy as np
import torch
from PIL import Image 

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

class InterpolateImage(torch.nn.Module):
    def __init__(self, data, device):
        super(InterpolateImage, self).__init__()
        self.data = data
        self.shape = self.data.shape
        self.data = torch.from_numpy(self.data).float().to(device)

    def forward(self, xs):
        with torch.no_grad():
            # Bilinearly filtered lookup from the image. Not super fast,
            # but less than ~20% of the overall runtime of this example.
            shape = self.shape

            xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()
            indices = xs.long()
            lerp_weights = xs - indices.float()

            x0 = indices[:, 0].clamp(min=0, max=shape[1]-1)
            y0 = indices[:, 1].clamp(min=0, max=shape[0]-1)
            x1 = (x0 + 1).clamp(max=shape[1]-1)
            y1 = (y0 + 1).clamp(max=shape[0]-1)

            return (
                self.data[y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
                self.data[y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
                self.data[y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
                self.data[y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
            )

def bgr_to_rgb(bgr_image):
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image

def rgb_to_bgr(rgb_image):
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image

def to_np(tensor) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError("Input should be a numpy array or a torch tensor")
    
def float_to_uint8(image):
    if isinstance(image, np.ndarray):
        return (np.clip(image,0.0, 1.0) * 255).astype(np.uint8)
    elif isinstance(image, torch.Tensor):
        return (torch.clamp(image, 0.0, 1.0) * 255.0).to(torch.uint8)
    else:
        raise TypeError("Input should be a numpy array or a torch tensor")
    
def uint8_to_float(image):
    if isinstance(image, np.ndarray):
        return image.astype(np.float32) / 255.0
    elif isinstance(image, torch.Tensor):
        return image.float() / 255.0
    else:
        raise TypeError("Input should be a numpy array or a torch tensor")
    

def write_image(file, image, image_order = "rgb"):
    if image_order == "bgr":
        cv2.imwrite(file, image)
    else:
        cv2.imwrite(file, rgb_to_bgr(image))

def write_exr(file, image, image_order = "rgb"):
    write_image(file,image, image_order)

def write_png(file, image, image_order = "rgb"):
    write_image(file,image, image_order)

def convert_to_grayscale(images):
    '''input (B, H, W, 3) or (H, W, 3)'''
    # Weights for the R, G, B channels
    weights = np.array([0.2989, 0.5870, 0.1140])
    
    # Use np.dot to multiply each pixel by the weights and sum the result
    grayscale_images = np.dot(images, weights)
    
    return grayscale_images


def read_exr(file, target_order = "rgb"):
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img = img[:,:,:3]
    if target_order == "bgr":
        return img
    else:
        return bgr_to_rgb(img)


def pil_float_image(image, order="rgb", linear=False):
    if linear:
        image = linear_to_srgb(image)

    if order == "bgr":
        return Image.fromarray(bgr_to_rgb(float_to_uint8(image)))
    else:
        return Image.fromarray(float_to_uint8(image))

def srgb_to_linear(img):
    limit = 0.04045
    if isinstance(img, np.ndarray):
        return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)
    elif isinstance(img, torch.Tensor):
        return torch.where(img > limit, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)
    else:
        raise TypeError("Input should be a numpy array or a torch tensor")

def linear_to_srgb(img):
    limit = 0.0031308
    if isinstance(img, np.ndarray):
        return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)
    elif isinstance(img, torch.Tensor):
        return torch.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)
    else:
        raise TypeError("Input should be a numpy array or a torch tensor")

def spherical_to_cartesian(theta, phi):
    '''
    input range:
        phi(-pi, pi) 
        theta(0, pi)
    '''
    return torch.stack([
        torch.sin(theta) * torch.cos(phi), # z-direction in CG coordinate
        torch.sin(theta) * torch.sin(phi), # x-direction in CG coordinate
        torch.cos(theta)                   # y-direction in CG coordinate
    ], dim=1)

def cartesian_to_spherical(data, device, normalize = True):
    '''
        data: (B, 3) CG coordinate
        return range (-pi, pi) (0, pi) or [0, 1) if normalize == true    
    '''
    y = data[:, 0]
    z = data[:, 1]
    x = data[:, 2]
    
    theta = torch.acos(z) 
    phi = torch.atan2(y, x)
    angle = torch.stack((phi, theta), dim=1 )

    if normalize:
        angle = (angle - torch.tensor([-torch.pi, 0], device=device)) * (1 / torch.tensor([2*torch.pi, torch.pi], device=device))

    return angle


def grid_sample_spherical_sapce(theta_l, theta_u, theta_n, phi_l, phi_u, phi_n):
    '''return torch.tensor in shape (theta_n * phi_n, 3)'''
    # Generate uniformly spaced values for theta and phi
    theta = torch.linspace(theta_l, theta_u, theta_n)  # Polar angle: 0 <= theta <= pi
    phi = torch.linspace(phi_l, phi_u, phi_n)  # Azimuthal angle: -pi <= phi < pi

    theta, phi = torch.meshgrid(theta, phi, indexing='ij')

    return spherical_to_cartesian(theta.reshape(-1), phi.reshape(-1))

def grid_sample_sphere(num_theta, num_phi):
    '''return torch.tensor in shape (num_theta * num_phi, 3)'''
    return grid_sample_spherical_sapce(0, np.pi, num_theta, np.pi, -np.pi, num_phi)

def grid_sample_hemisphere(num_theta, num_phi):
    '''return torch.tensor in shape (num_theta * num_phi, 3)'''
    return grid_sample_spherical_sapce(0, np.pi / 2, num_theta, np.pi, -np.pi, num_phi)

def random_sample_sphere(n):
    '''return torch.tensor spherical(theta, phi) and cartesian in shape (n, 2), (n, 3)'''
    samples = torch.rand((n, 2)) # theta, phi
    samples[:, 0] = 2 * torch.acos(torch.sqrt(1 - samples[:, 0])) # theta
    samples[:, 1] = 2 * np.pi * (samples[:, 1]-0.5) # phi

    return samples, spherical_to_cartesian(samples[:, 0], samples[:, 1])


def distance(X):
    '''X shape (N, x) return shape (N,)'''
    return torch.sqrt(torch.sum(X ** 2, dim=1))

def dot(X, Y):
    '''
        - X, Y in shape (B, N), do dot prodcut for each element return (B,1) 
        - X in shape (B, C, N),
            Y in shape (B, N)
            do dot prodcut for each element return (B, C)
    '''
    if len(X.shape) == 2:
        '''X, Y in shape (B, N), do dot prodcut for each element return (B,1)'''
        X = X.unsqueeze(1).float()
        Y = Y.unsqueeze(2).float()
        result = torch.bmm(X, Y)
        result = result.view(-1, 1)
        return result
    else:
        '''
        X in shape (B, C, N),
        Y in shape (B, N)
        do dot prodcut for each element return (B, C)
        '''
        # First, add a dimension to tensor2 to make it (B, N, 1)
        Y = Y.unsqueeze(2)  # Now tensor2 is (B, N, 1)

        # Perform batch matrix multiplication
        result = torch.bmm(X, Y)  # Result is (B, C, 1)

        # Remove the last dimension to get the final result of shape (B, C)
        result = result.squeeze(2)  # Now result is (B, C)

        return result

def eval_gaussian(dir, color, mean, scale):
    '''
        dir (B, 3) [-1, 1]
        color (B, N, 3)
        mean (B, N, 3)
        scale (B, N)

        return (B, 3)
    '''
    return (color * torch.exp(scale * (dot(mean, dir) -1) ).unsqueeze(2)).sum(dim=1)

def lerp(a, b, t):
    return a + (b - a) * t

def str2array(value:str):
    return np.array([float(v) for v in value.split(" ")])
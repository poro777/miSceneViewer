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

def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    '''copy from https://github.com/limacv/RGB_HSV_HSL'''
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    '''copy from https://github.com/limacv/RGB_HSV_HSL'''
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb

def adjust_brightness_hsv(img:torch.Tensor, lum):
    # Convert to HSV
    hsv = rgb_to_hsv(img.permute(2, 0, 1).unsqueeze(0))
    
    # Adjust the brightness (value channel)
    hsv[:, 2] = lum
    
    # Convert back to RGB
    rgb_adjusted = hsv_to_rgb(hsv)
    
    return rgb_adjusted[0].permute(1, 2, 0)

def tonemap_photographic(hdr_image, key = 0, adaptive:bool = False):
    '''
        key: the key of the image after applying the tone mapping.
        adaptive: calculate the key of the input image.
        reference Reinhard, Erik, et al. "Photographic tone reproduction for digital images."
    "'''
    alpha = 0.18 * (2 ** key)

    # Ensure the input HDR image is a floating point tensor
    if hdr_image.dtype != torch.float32 and hdr_image.dtype != torch.float64:
        raise ValueError("Input HDR image must be a floating point tensor")

    # Calculate luminance (assuming input is in RGB)
    luminance: torch.Tensor = 0.2126 * hdr_image[:, :, 0] + 0.7152 * hdr_image[:, :, 1] + 0.0722 * hdr_image[:, :, 2]

    Lw = torch.exp(torch.log(luminance + 1e-8).mean()) if adaptive else 0.1

    L = alpha / Lw * luminance

    Lwhite = torch.max(luminance)
    Ld = L * (1 + L / (Lwhite**2)) / (1 + L)

    tonemapped_image = adjust_brightness_hsv(hdr_image, Ld)
    tonemapped_image = torch.clamp(tonemapped_image, 0, 1)

    return tonemapped_image

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
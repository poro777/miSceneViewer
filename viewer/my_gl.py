from PIL import Image
import numpy as np
import OpenGL.GL as gl
import torch
import pygame

try:
    import pycuda.gl as cudagl
    import pycuda.autoinit
    import pycuda.driver as cuda
    withPyCuda = True
except:
    withPyCuda = False

# Load texture using PIL
def delete_gl_texture(texture_id):
    gl.glDeleteTextures([texture_id])

def create_gl_texture(width, height):
    img_data = Image.fromarray(np.zeros((height,width,4)).astype(np.uint8)).tobytes("raw", "RGBA", 0, -1)

    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)

    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0,gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data)
    return texture_id

def create_map_texture(width, height):
    texture_id = create_gl_texture(width, height)
    cuda_gl_texture = cudagl.RegisteredImage(int(texture_id), gl.GL_TEXTURE_2D, cudagl.graphics_map_flags.WRITE_DISCARD)
    return texture_id, cuda_gl_texture

def render_texture(texture_id):
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glBegin(gl.GL_QUADS)

    gl.glTexCoord2f(0.0, 1.0)
    gl.glVertex3f(-1.0, -1.0, 0.0)

    gl.glTexCoord2f(1.0, 1.0)
    gl.glVertex3f(1.0, -1.0, 0.0)

    gl.glTexCoord2f(1.0, 0.0)
    gl.glVertex3f(1.0, 1.0, 0.0)

    gl.glTexCoord2f(0.0, 0.0)
    gl.glVertex3f(-1.0, 1.0, 0.0)

    gl.glEnd()


def copy_tensor_to_texture(image: torch.Tensor, texture_id, cuda_gl_texture):
    assert withPyCuda == True
    alpha_channel = torch.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype, device=image.device)
    tensor = torch.cat((image, alpha_channel), dim=2)

    tensor = tensor.byte().contiguous() # convert to ByteTensor

    # copy from torch into buffer
    mapping = cuda_gl_texture.map()
    ary = mapping.array(0,0)
    
    cpy = cuda.Memcpy2D()
    cpy.set_src_device(tensor.data_ptr())
    cpy.set_dst_array(ary)

    # Ensure width_in_bytes is correct
    cpy.width_in_bytes = tensor.shape[1] * tensor.shape[2] * tensor.element_size()
    cpy.src_pitch = cpy.width_in_bytes  # Tensor row pitch in bytes
    cpy.dst_pitch = cpy.width_in_bytes  # Destination array row pitch in bytes
    cpy.height = tensor.shape[0]  # Number of rows to copy

    cpy(aligned=True)
    torch.cuda.synchronize()
    mapping.unmap()


def render_cpu_array(image: np.ndarray, w, h):
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    surface = pygame.surfarray.make_surface(image.swapaxes(0,1))
    surface = pygame.transform.scale(surface, (w, h))
    textData = pygame.image.tostring(surface, "RGBA", True)
    gl.glWindowPos2i(0, 0)
    gl.glDrawPixels(w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, textData)
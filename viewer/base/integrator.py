from abc import ABC
from typing import Dict
import imgui
import torch

class Integrator(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.object = None

    def update(self, *arg, **kwarg):
        raise NotImplementedError
    
    def name(self):
        return "Custom"

    def postprocess(self, images, sys_info):
        '''return torch.tensor'''
        raise NotImplementedError
    
    def render(self, SPP):
        '''return data dict'''
        raise NotImplementedError
    
    def save_image(self, images):
        return {
            "img": None # set None to ues current frame as output image or self.postprocess(images, sys_info)
        }

    def gui(self):
        pass

    def resize(self, width, height, fov):
        raise NotImplementedError

    def setCameraPose(self, to_world):
        raise NotImplementedError

    def getOutput(self, SPP, sys_info):
        '''return torch.tensor'''
        return self.postprocess(self.render(SPP), sys_info)
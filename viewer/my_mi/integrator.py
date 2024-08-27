import numpy as np
import mitsuba as mi
import drjit as dr
import torch
from viewer.common import to_np
from viewer.common import *
from abc import ABC
from typing import Dict
import imgui
import torch
from viewer.my_mi.scene import mitsuba_scene
from viewer.my_mi.sensor import mitsiba_sensor

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
    
    def render(self):
        '''return data dict'''
        raise NotImplementedError
    
    def save_image(self, images):
        return {
            "img": None # set None to ues current frame as output image or self.postprocess(images, sys_info)
        }

    def gui(self):
        pass

    def getOutput(self, SPP, sys_info):
        '''return torch.tensor'''
        return self.postprocess(self.render(SPP), sys_info)

    
class mitsuba_Integrator(Integrator):
    def __init__(self) -> None:
        super().__init__()
    def update(self, depth, *arg, **kwarg):
        raise NotImplementedError
      
    def render(self, scene:mitsuba_scene, sensor:mitsiba_sensor, SPP, *arg, **kwarg):
        with dr.suspend_grad():
            result:mi.TensorXf = mi.render(scene.object, spp=SPP, integrator = self.object, sensor=sensor.object ,seed=np.random.randint(0,999))
            return {"output": result.torch()}
        
    def postprocess(self, images, sys_info):
        return images["output"]
    
class path_Integrator(mitsuba_Integrator):
    def __init__(self) -> None:
        super().__init__()
    
    def name(self):
        return "Path"
        
    def update(self, depth, *arg, **kwarg):
        self.object =  mi.load_dict({
                'type': 'path',
                'max_depth': depth
            })

class ptrace_Integrator(mitsuba_Integrator):
    def __init__(self) -> None:
        super().__init__()
    
    def name(self):
        return "Ptracer"
        
    def update(self, depth, *arg, **kwarg):
        self.object =  mi.load_dict({
                'type' : "ptracer",
                'max_depth': depth
            })

class path_info_Integrator(mitsuba_Integrator):
    def __init__(self) -> None:
        super().__init__()
        self.itemsName = ["Image", "Position", "Albedo", "Normal"]
        self.items = ["img", "pos", "alb", "nor"]
        self.showItem = 0

    def name(self):
        return "PathInfo"
        
    def update(self, depth, *arg, **kwarg):
        self.object =  mi.load_dict({
            'type': 'aov',
            'aovs': 'pos:position,alb:albedo,nor:sh_normal',
            'my_image': {
                'type': 'path',
                'max_depth': depth
            }
        })

    def render(self, scene:mitsuba_scene, sensor:mitsiba_sensor, SPP):
        images = mitsuba_Integrator.render(self, scene, sensor, SPP)["output"]
        img = images[:,:,0:3]
        pos = images[:,:,4:7]
        alb = images[:,:,7:10]
        nor = images[:,:,10:13]
        return {
            "img": img,
            "pos": pos,
            "alb": alb,
            "nor": nor
        }  
    
    def postprocess(self, images, sys_info):
        return images[self.items[self.showItem]]

    def gui(self):
        changed, self.showItem = imgui.combo("Render result", self.showItem, self.itemsName)

    def save_image(self, images):
        images = {k:to_np(v) for k,v in images.items()}
        images["img"] = None
        return images
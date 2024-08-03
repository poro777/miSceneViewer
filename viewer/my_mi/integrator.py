import numpy as np
import mitsuba as mi
import drjit as dr
from viewer.base.integrator import *
import torch
from viewer.common import to_np
from viewer.common import *

    
class mitsuba_Integrator(Integrator):
    def __init__(self, scene) -> None:
        super().__init__()
        self.scene = scene

    def resize(self, width, height, fov):
        self.width = width
        self.height = height
        
        self.sensor = mi.load_dict({
            'type': 'perspective',
            'fov': fov,
            'to_world': mi.ScalarTransform4f(),
            'sampler_id':{
                'type': 'independent'
            },
            'film_id': {
                'type': 'hdrfilm',
                'width': self.width,
                'height': self.height,
                'pixel_format': 'rgba',
                "filter": {"type": "box"},
            }
        })
        self.params = mi.traverse(self.sensor)

    def setCameraPose(self, to_world):
        self.params['to_world'] =  to_world
        self.params.update()
    
    def render(self, SPP):
        with dr.suspend_grad():
            result:mi.TensorXf = mi.render(self.scene, spp=SPP, integrator = self.object, sensor=self.sensor ,seed=np.random.randint(0,999))
            return {"output": result.torch()}
        
    def postprocess(self, images, sys_info):
        return images["output"]
    
class path_Integrator(mitsuba_Integrator):
    def __init__(self, scene) -> None:
        super().__init__(scene)
    
    def name(self):
        return "Path"
        
    def update(self, depth):
        self.object =  mi.load_dict({
                'type': 'path',
                'max_depth': depth
            })

class ptrace_Integrator(mitsuba_Integrator):
    def __init__(self, scene) -> None:
        super().__init__(scene)
    
    def name(self):
        return "Ptracer"
        
    def update(self, depth):
        self.object =  mi.load_dict({
                'type' : "ptracer",
                'max_depth': depth
            })

class path_info_Integrator(mitsuba_Integrator):
    def __init__(self, scene) -> None:
        super().__init__(scene)
        
    def name(self):
        return "Path"
        
    def update(self, depth):
        self.object =  mi.load_dict({
            'type': 'aov',
            'aovs': 'pos:position,alb:albedo,nor:sh_normal',
            'my_image': {
                'type': 'path',
                'max_depth': depth
            }
        })

    def render(self, SPP):
        images = mitsuba_Integrator.render(self, SPP)["output"]
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
        return images["img"]

    def save_image(self, images):
        images = {k:to_np(v) for k,v in images.items()}
        images["img"] = None
        return images
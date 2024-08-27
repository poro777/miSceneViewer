import numpy as np
import mitsuba as mi
import drjit as dr
import torch
import imgui

class mitsiba_sensor:
    def __init__(self) -> None:
        self.object = None

    def gui(self):
        pass

    def resize(self, width, height, fov):
        self.width = width
        self.height = height
        
        self.object = mi.load_dict({
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
        self.params = mi.traverse(self.object)
    
    def setCameraPose(self, to_world):
        '''set camera pose to to_world matrix. 
        - to_woald matrix can get from mitsuba_scene.getCameraMatrix()'''

        if self.object is None:
            raise ValueError("Call resize(...) before setCameraPose()")
        
        self.params['to_world'] =  to_world
        self.params.update()
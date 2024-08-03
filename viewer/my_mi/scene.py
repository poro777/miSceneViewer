from viewer.base.scene import Scene
import mitsuba as mi
from viewer.common import cartesian_to_spherical
import torch
import numpy as np
from viewer.animation import *
import drjit as dr

class mitsuba_scene(Scene):
    def __init__(self, scene_xml, animation_xml = None) -> None:
        super().__init__()
        
        self.object = mi.load_file(scene_xml)
        self.animation = load_animation(animation_xml) if (animation_xml is not None) else {}
        self.animated_shape_initial_state = {}

        self.camera_to_world = None
        self.params = mi.traverse(self.object)

        
        self.sensors = [sensor.id() for sensor in self.object.sensors()]

        # use first camera
        if len(self.sensors) > 0:
            self.camera = self.sensors[0]
            if f'{self.sensors[0]}.to_world' in self.params:
                self.camera_to_world = self.params[f'{self.sensors[0]}.to_world']
            elif 'PerspectiveCamera.to_world' in self.params:
                self.camera_to_world = self.params['PerspectiveCamera.to_world']
            else:
                print(f"Cannot load default sensor {self.camera} in file")

        self.time = 0
        for name, animation in self.animation.items():
            self.time = max(self.time, animation.endTiem)
            if f'{name}.vertex_positions' in self.params:  # shape
                v = dr.unravel(mi.cuda_ad_rgb.Point3f, self.params[f'{name}.vertex_positions'])
                m0 = mi.cuda_ad_rgb.Transform4f(animation.getMatrix(0.0))
                self.animated_shape_initial_state[name] = m0.inverse() @ v
            elif f'{name}.to_world' in self.params:     # camera
                self.camera_to_world = self.params[f'{name}.to_world']
                self.camera  = name
                m0 = mi.cuda_ad_rgb.Transform4f(animation.getMatrix(0.0))
                cameraMatrix = m0.inverse() @ self.camera_to_world
                self.animated_shape_initial_state[name] = cameraMatrix

        
    def scene_bbox_info(self):
        bbox = self.object.bbox()
        bbox_min = list(bbox.min)
        bbox_scale = list(bbox.max - bbox.min)
        return bbox_min, bbox_scale

    def _matrix2camera(self, toWorld):
        origin = np.array(toWorld @ mi.Point3f(0,0,0))
        direction = np.array(toWorld @ mi.Vector3f(0,0,1))
        return origin[0], cartesian_to_spherical(torch.tensor(direction), "cpu", False)[0].numpy()
    
    def setTime(self, time):
        for name, animation in self.animation.items():
            if f'{name}.vertex_positions' in self.params:
                v = self.animated_shape_initial_state[name]
                m = mi.cuda_ad_rgb.Transform4f(animation.getMatrix(time))
                self.params[f'{name}.vertex_positions'] = dr.ravel(m @ v)
        
        self.params.update()
    
    def getMaxTime(self):
        return self.time
    
    def getCamera(self, time):
        if self.camera in self.animation:
            animation = self.animation[self.camera]
            camera_matrix = self.animated_shape_initial_state[self.camera]
            m = mi.cuda_ad_rgb.Transform4f(animation.getMatrix(time, applyScale=False))
            return self._matrix2camera(m @ camera_matrix)
        else:
            return self.get_default_camera()

    def get_default_camera(self):
        if self.camera_to_world is not None:
            return self._matrix2camera(self.camera_to_world)
        else:
            return np.array([0,0,0], dtype=float), np.array([0, np.pi/2])
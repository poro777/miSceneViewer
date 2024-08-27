import mitsuba as mi
from viewer.common import cartesian_to_spherical
import torch
import numpy as np
from viewer.animation import *
import drjit as dr
from viewer.my_mi.sensor import *

def matrix2camera(toWorld):
        origin = np.array(toWorld @ mi.Point3f(0,0,0))
        direction = np.array(toWorld @ mi.Vector3f(0,0,1))
        return origin[0], cartesian_to_spherical(torch.tensor(direction), "cpu", False)[0].numpy()

def camera2matrix(origin, direciion):
    def getTarget():
        sin1 = np.sin(direciion[1])
        return origin + np.array([sin1 * np.sin(direciion[0] ), np.cos(direciion[1]), sin1 * np.cos(direciion[0] )])
    
    return mi.ScalarTransform4f.look_at(origin, getTarget(), np.array([0,1,0.0], dtype=float))

class mitsuba_scene:
    def __init__(self, scene_xml, animation_xml = None) -> None:
        super().__init__()
        
        self.object = mi.load_file(scene_xml)
        self.camera_id = None
        self.params = mi.traverse(self.object)

        self.showCamera = 0

        self.time = 0
        self.animation = load_animation(animation_xml) if (animation_xml is not None) else {}
        self.animated_shape_initial_state = {} # local frame
 
        self.cameras_id:list[str] = [sensor.id() for sensor in self.object.sensors()]
        self.cameras_to_world = {}

        unname_id = 0
        for i, camera_id in enumerate(self.cameras_id):
            if '_unnamed' in camera_id:
                new_camera_id = 'PerspectiveCamera' if unname_id == 0 else f'PerspectiveCamera_{unname_id}'
                unname_id += 1
                self.cameras_id[i] = new_camera_id
                camera_id = new_camera_id

            self.cameras_to_world[camera_id] = self.params[f'{camera_id}.to_world']

        # use first camera
        self.setCamera(0)

        
        # assume all animated object has id (not unnamed)
        for id, animation in self.animation.items():
            self.time = max(self.time, animation.endTiem)
            if f'{id}.vertex_positions' in self.params:  # shape
                v = dr.unravel(mi.cuda_ad_rgb.Point3f, self.params[f'{id}.vertex_positions'])
                m0 = mi.cuda_ad_rgb.Transform4f(animation.getMatrix(0.0))
                self.animated_shape_initial_state[id] = m0.inverse() @ v
            elif f'{id}.to_world' in self.params:     # camera
                m0 = mi.cuda_ad_rgb.Transform4f(animation.getMatrix(0.0))
                cameraMatrix = m0.inverse() @ self.cameras_to_world[id]
                self.animated_shape_initial_state[id] = cameraMatrix

        
    def scene_bbox_info(self):
        bbox = self.object.bbox()
        bbox_min = list(bbox.min)
        bbox_scale = list(bbox.max - bbox.min)
        return bbox_min, bbox_scale

    def setTime(self, time):
        '''set animated object's pose at time.(camera not included)'''
        for name, animation in self.animation.items():
            if f'{name}.vertex_positions' in self.params:
                v = self.animated_shape_initial_state[name]
                m = mi.cuda_ad_rgb.Transform4f(animation.getMatrix(time))
                self.params[f'{name}.vertex_positions'] = dr.ravel(m @ v)
        
        self.params.update()
    
    def getMaxTime(self):
        return self.time
    
    def setCamera(self, index):
        '''Set current activate camera by index. getCamera() will return corresponding camera pose'''
        if index < 0 or index >= len(self.cameras_id):
            return
        self.camera_id = self.cameras_id[index]

    def getCameraMatrix(self, time, index = None):
        '''return activated camera pose's to_world matrix. This pose will apply animation (if available)'''
        return camera2matrix(*self.getCamera(time, index))
        
    def getCamera(self, time, index = None):
        '''return activated camera pose (position, direction). This pose will apply animation (if available) '''
        if index is not None:
            self.setCamera(index)

        if self.camera_id in self.animation:
            animation = self.animation[self.camera_id]
            camera_matrix = self.animated_shape_initial_state[self.camera_id]
            m = mi.cuda_ad_rgb.Transform4f(animation.getMatrix(time, applyScale=False))
            return matrix2camera(m @ camera_matrix)
        else:
            return self.get_default_camera()

    def get_default_camera(self, index = None):
        '''return activated camera pose (position, direction) without apply animation (t = 0).'''
        if index is not None:
            self.setCamera(index)

        if self.camera_id is not None:
            toWorld = self.cameras_to_world[self.camera_id]
            return matrix2camera(toWorld)
        else:
            return np.array([0,0,0], dtype=float), np.array([0, np.pi/2])
    
    def gui(self):
        pass
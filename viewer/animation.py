import transformations
import numpy as np
from typing import Dict
import xml.etree.ElementTree as ET
import os
from viewer.common import lerp, str2array

def scale_matrix_from_array(scaling):
    m = transformations.identity_matrix()
    m[0, 0] = scaling[0]
    m[1, 1] = scaling[1]
    m[2, 2] = scaling[2]
    return m

def transfromMatrix(pos, rot, scale):
    return transformations.translation_matrix(pos) @ transformations.quaternion_matrix(rot) @ scale_matrix_from_array(scale)

def find_interval_index(arr, x):
    ''' binary search interval'''
    if x <= arr[0] or x >= arr[-1]:
        return -1  # x is not within any interval in the array
    
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] <= x < arr[mid + 1]:
            return mid
        elif x < arr[mid]:
            right = mid - 1
        else:
            left = mid + 1
            
    return -1  # x is not within any interval in the array

class animation:
    def __init__(self, N) -> None:
        self.length = N
        self.pos = np.zeros((N, 3))
        self.quat = np.zeros((N, 4))
        self.scaling = np.zeros((N, 3))
        self.time = np.zeros(N)

        self.beginTime = 0
        self.endTiem = 0

        self.to_world = transformations.identity_matrix()

    def setKey(self, k: int, pos: np.ndarray, quat: np.ndarray, scaling: np.ndarray, time: float):
        self.pos[k] = pos if (pos is not None) else self.pos[0]
        self.quat[k] = quat if (quat is not None) else self.quat[0]
        self.scaling[k] = scaling if (scaling is not None) else self.scaling[0]
        self.time[k] = time 

    def getMatrix(self, time: float, applyTranslate = True, applyRotate = True, applyScale = True):
        index = find_interval_index(self.time, time)
        if index == -1:
            if time <= self.time[0]:
                pos = self.pos[0]
                rot = self.quat[0]
                scale = self.scaling[0]
            else:
                pos = self.pos[-1]
                rot = self.quat[-1]
                scale = self.scaling[-1]
        else:
            t = (time - self.time[index]) / (self.time[index + 1] - self.time[index])
            pos = lerp(self.pos[index], self.pos[index + 1], t)
            rot = transformations.quaternion_slerp(self.quat[index], self.quat[index + 1], t)
            scale = lerp(self.scaling[index], self.scaling[index + 1], t)

        if not applyTranslate:
            pos = [0,0,0]
        if not applyRotate:
            rot = [0,0,0,0]
        if not applyScale:
            scale = [1,1,1]

        return self.to_world @ transfromMatrix(pos, rot, scale)

        

def load_animation(file: str):
    ''' parse animation.xml 
        return Dict[str, animation]
    '''
    if not os.path.exists(file):
        return {}
    
    tree = ET.parse(file) # From file
    root = tree.getroot()

    animations: Dict[str, animation] = {}

    for shapes in root.findall("shape"):
        # Assume the order is the same as in the file

        name = shapes.attrib['id']
        ref = shapes.find("ref")
        if ref != None:
            # point to reference item
            ani = animations[ref.attrib['value']]
        else:
            length = int(shapes.attrib['max'])
            ani = animation(length)
            for key in shapes.findall("key"):
                k = int(key.attrib['value'])
                time = float(key.attrib['time'])
                transform = key.find("transform")
                pos = str2array(transform.attrib['position']) if 'position' in transform.attrib else None
                rot = str2array(transform.attrib['rotation']) if 'rotation' in transform.attrib else None
                scale = str2array(transform.attrib['scaling']) if 'scaling' in transform.attrib else None
                ani.setKey(k, pos, rot, scale, time)
            transform = shapes.find("transform")
            if transform is not None:
                matrix = transform.find("matrix").attrib['value']
                ani.to_world = str2array(matrix).reshape((4,4))
            ani.beginTime = ani.time[0]
            ani.endTiem = ani.time[-1]

        animations[name] = ani
    
    return animations

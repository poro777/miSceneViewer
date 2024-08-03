import json
import mitsuba as mi

from tqdm import tqdm
from viewer.execution_time import *
from viewer.common import *

class Dataset():
    def __init__(self, image_dir, size = 0) -> None:
        self.info = []
        self.image_dir = image_dir
        self.size = [size, size]
        self._data_cache = None
        self._images_cache = {}

        self.max_id = -1

        if not os.path.exists(self.image_dir):
            os.mkdir(self.image_dir)
                
        self.load_info()

    def _load_image(self, id, type):
        image_file = os.path.join(self.image_dir, f"output_{id}_{type}.exr") 
        if not os.path.exists(image_file):
            print(f"No such file : {image_file}")
            return np.zeros(self.size)
        
        img = read_exr(image_file)
        if img is None:
            print(f"Can not read file : {image_file}")
            return np.zeros(self.size)
        return cv2.resize(img, self.size)

    def load_id(self, id_info):
        '''
        - return (H , W , C)
        - img, pos, ori, alb, nor
        '''
        id = id_info["id"]
        origin = id_info["origin"]

        if id in self._images_cache:
            return self._images_cache[id]
        
        img = self._load_image(id, "img")
        pos = self._load_image(id, "pos")
        alb = self._load_image(id, "alb")
        nor = self._load_image(id, "nor")

        ori = np.tile(origin, self.size + [1])

        self._images_cache[id] = (img, pos, ori, alb, nor)
        
        # return (H , W , C)
        return img, pos, ori, alb, nor
    
    def write_outputs(self, output:dict, sys_info, also_save_img_to_png = True):
        '''outputs in linear space'''
        self.max_id += 1   
        id = self.max_id

        self.store_info(id, sys_info['fov'], sys_info['to_world'])

        for postfix, image in output.items():
            path = os.path.join(self.image_dir, f"output_{id}_{postfix}.exr") 
            write_exr(path, image)

            if postfix == "img" and also_save_img_to_png:
                path = os.path.join(self.image_dir, f"output_{id}_{postfix}.png") 
                write_png(path, float_to_uint8(linear_to_srgb(image)))

    def write_images(self, image, sys_info):
        '''image in uint8'''
        self.max_id += 1   
        id = self.max_id

        self.store_info(id, sys_info['fov'], sys_info['to_world'])
        postfix = "img"

        path = os.path.join(self.image_dir, f"output_{id}_{postfix}.png") 
        write_png(path, image)

    def store_info(self, id, fov, to_world):
        filePath = os.path.join(self.image_dir, "info.txt")
        self.sensors[str(id)] = {
            'fov': fov,
            'to_world': [list(i.astype(float)) for i in np.array(to_world.matrix)]
        }

        self.sensors = dict(sorted(self.sensors.items(), key=lambda item: int(item[0])))
        with open(filePath, 'w') as file:
            json.dump(self.sensors, file, indent=4)

    def load_info(self):
        self.sensors = {}
        self.n_sensor = 0

        filePath = os.path.join(self.image_dir, "info.txt")
        if not os.path.exists(filePath):
            return
        
        with open(filePath) as f:
            sensors = dict(json.load(f))

        self.sensors = dict(sorted(sensors.items(), key=lambda item: int(item[0])))
        self.n_sensor = len(self.sensors)

        for i, sensor in self.sensors.items():
            m = sensor['to_world']
            fov, to_world, origin = sensor["fov"], mi.ScalarTransform4f(m), np.array([m[0][3], m[1][3], m[2][3]])

            parsed_dict = {
                'id': i,
                'to_world': to_world,
                'fov': fov,
                'origin': origin
            }
            self.info.append(parsed_dict)
            self.max_id = max(int(i), self.max_id)

from abc import ABC

class Scene(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.object = None

    def scene_bbox_info(self):
        raise NotImplementedError
    
    def getMaxTime(self):
        return 0.0
    
    def getCamera(self, time):
        raise NotImplementedError
    
    def get_default_camera(self):
        raise NotImplementedError
    
    def setTime(self, time):
        raise NotImplementedError
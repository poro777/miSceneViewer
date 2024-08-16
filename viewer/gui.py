from dataclasses import dataclass
import imgui

def radio_button(list_of_name, selected_index):
    changed = False
    for i, name in enumerate(list_of_name):
        if imgui.radio_button(name, selected_index == i):
            changed = True
            selected_index = i
        imgui.same_line()
    imgui.new_line()
    return changed, selected_index

@dataclass
class Window:
    windowName = "Viewer"
    windowWidth = 768
    windowHeight = 768
    auto_resize = True

@dataclass
class Sensor:
    SPP = 32
    resoultion = 9

    limit = (2 ** 26)
    depth = 6
    fov = 45

@dataclass
class Once:
    snapshot_output = False
    eval_flip = False
    save_output = False # save integraor output

    def reset(self):
        self.snapshot_output = False
        self.eval_flip = False
        self.save_output = False

@dataclass
class Speed:
    move = 0.15
    view = 0.003

@dataclass
class Animation:
    time = 0
    animate_camera = True

    def gui(self, maxTime):
        _, self.animate_camera = imgui.checkbox("Camera", self.animate_camera)
        changed, self.time = imgui.slider_int("Time", self.time, 0, maxTime)
        return changed

@dataclass
class Progress:
    enable = False
    t = 0

    def gui(self):
        changed, self.enable = imgui.checkbox("Progress", self.enable)
        imgui.same_line()
        imgui.text(f"t: {self.t}")
        return changed

@dataclass
class Flip:
    display = False
    
    selected_type = 0
    selected_range = 0
    error = None
    name_list = ["Flip", "GT", "Test"]
    range_list = ["HDR", "LDR"]

    def gui(self):
        changed, self.display = imgui.checkbox("Show flip", self.display)
        imgui.text(f"Error: {self.error}")
        _, self.selected_range = imgui.combo("Range", self.selected_range, self.range_list)
        _, self.selected_type = radio_button(self.name_list, self.selected_type)
        return changed
    
@dataclass
class Variable:
    window = Window()
    sensor = Sensor()
    once = Once()
    speed = Speed()
    animation = Animation()
    progress = Progress()
    flip = Flip()

    stop = False
    fps_guard = True

    selected_integrator = 0
    
    selected_dataset_sensor = -1  # Set the sensor view to the view in the dataset
    selected_scene_sensor = 0   # activate sensor (reset view or apply animation)

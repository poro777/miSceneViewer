import drjit as dr
import mitsuba as mi
import numpy as np

import torch

import pygame
from imgui.integrations.pygame import PygameRenderer
import OpenGL.GL as gl
import imgui

from dataclasses import dataclass
from typing import List

mi.set_variant('cuda_ad_rgb')

from viewer.common import *
from viewer.execution_time import *
from viewer.base import *
from viewer.dataset import *
from viewer.metric import *

@dataclass
class variable:
    windowName = "Viewer"
    windowWidth = 768
    windowHeight = 768

    auto_resize = True
    SPP = 32
    resoultion = 9

    limit = (2 ** 26)

    save_image = False
    snapshot_output = False

    stop = False
    
    MOVE_SPEED = 0.15
    VIEW_SPEED = 0.003
    depth = 6
    selected_radio = 0
    fov = 45
    progress = False
    progress_t = 0

    save = False

    store_gt = False
    show_flip = False
    eval_flip = False
    show_flip_type = 0

    sensor_id = -1

    fps_guard = True

    time = 0
    animate_camera = True

def load_surface(image, resize = None):
    '''resize (W, H)'''
    surface = pygame.surfarray.make_surface(to_np(image).swapaxes(0,1))
    if resize is not None:
        surface = pygame.transform.scale(surface, resize)
    return surface

def load_texture(textureSurface):

    textureData = pygame.image.tostring(textureSurface, "RGBA", False)

    width = textureSurface.get_width()
    height = textureSurface.get_height()

    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA,
                    gl.GL_UNSIGNED_BYTE, textureData)

    return texture, width, height


class render_system:
    Y = np.array([0,1,0]).astype(float)
    BLACK = torch.zeros((1,1,3)).cuda()
    def __init__(self, scene: Scene, origin = None, angle = None, dataset:Dataset = None, var:variable = None, integrators:List[Integrator] = None, snapshot:Dataset = None,) -> None:
        self.var = var if var is not None else variable()
        self.clock = pygame.time.Clock()

        self.scene = scene
        self.origin, self.angle = self.scene.get_default_camera()
        if origin is not None:
            self.origin = np.array(origin).astype(float)
        if angle is not None:
            self.angle = np.array(angle).astype(float)

        self.width = 2**self.var.resoultion
        self.height = self.width
        self.draging = False
        self.progress_image = None

        
        self.integrator_list : List[Integrator] = []

        if integrators is not None:
            self.integrator_list += integrators

        self.change_integrator(0)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # reset progress image if view change
        self.view_change = False
        
        self.app_live = 0
        self.waitingTimes = 0
        self.dataset = dataset
        self.flip = flipError()

        # camera to world
        self.to_world = None

        self.already_init = False

        self.snapshot = snapshot

    def change_integrator(self, id):
        assert(id < len(self.integrator_list) and id >= 0)

        self.integrator : Integrator = self.integrator_list[id]
        self.integrator.update(self.var.depth)
        self.integrator.resize(self.width, self.height, self.var.fov)

    def getDirection(self):
        angle = self.angle
        sin1 = np.sin(angle[1])
        return np.array([sin1 * np.sin(angle[0] ),
                        np.cos(angle[1]),
                        sin1 * np.cos(angle[0] )])

    def getTarget(self):
        return self.origin + self.getDirection()

    def scene_bbox_info(self):
        return self.scene.scene_bbox_info()
    
    @measure_execution_time
    def render(self, SPP = 32):
        with torch.no_grad():
            if self.var.sensor_id == -1 or self.to_world is None:
                self.to_world = mi.ScalarTransform4f.look_at(self.origin, self.getTarget(), render_system.Y)

            self.integrator.setCameraPose(self.to_world)
            return self.integrator.render(SPP)

    def setResoultion(self):
        '''call when resoultion or window size change'''
        ratio = self.var.windowHeight / self.var.windowWidth if self.var.auto_resize else 1.0
        self.width = min(2 ** self.var.resoultion, self.var.windowWidth)
        self.height = int(self.width * ratio)
        self.integrator.resize(self.width, self.height, self.var.fov)
        self.sample_guard()
        self.view_change = True

    def sample_guard(self):
        '''prevent low FPS'''
        if (self.var.SPP * self.height * self.width) > self.var.limit:
            self.var.stop = True

    def gui_frame(self, gui):
        imgui.new_frame()
        
        self.clock.tick()
        
        is_expand, show_custom_window = imgui.begin("Custom window", True)
        if is_expand:
            # information
            self.fps = self.clock.get_fps()
            if self.fps < 3 and self.app_live > 30 and self.var.fps_guard:
                self.waitingTimes += 1
                if self.waitingTimes > 5:
                    self.var.stop = True
            else:
                self.waitingTimes = 0

            imgui.text(f"FPS: {self.fps}")
            imgui.text(f"Window width:{self.var.windowWidth} height:{self.var.windowHeight}")
            imgui.text(f"Image width:{self.width} height:{self.height}")
            imgui.text(f"Position: {np.round(self.origin, 3)}")
            imgui.text(f"Direction: {np.round(self.angle, 3)}")

            changed, self.var.stop = imgui.checkbox("Stop", self.var.stop)
            if changed: self.sample_guard()

            if imgui.tree_node("Sensor"):
                resoultion_changed, self.var.auto_resize = imgui.checkbox("Fit to Window", self.var.auto_resize)

                changed, self.var.resoultion = imgui.slider_int("resoultion", self.var.resoultion, 5, 11)
                resoultion_changed = resoultion_changed or changed
                
                
                _, self.var.SPP = imgui.slider_int("SPP", self.var.SPP, 1, 256)

                depth_change, self.var.depth = imgui.slider_int("depth", self.var.depth, 1, 10)

                fov_changed, self.var.fov = imgui.slider_int("fov", self.var.fov, 10, 90)

                if self.dataset is not None:
                    sensor_changed, self.var.sensor_id = imgui.slider_int("Sensor", self.var.sensor_id, -1, self.dataset.n_sensor-1)
                    if sensor_changed:
                        fov_changed = True
                        if self.var.sensor_id != -1:  # default
                            info = self.dataset.info[self.var.sensor_id]
                            self.var.fov, self.to_world, self.origin = info['fov'], info['to_world'], info['origin']

                if depth_change:
                    self.integrator.update(self.var.depth)

                if resoultion_changed or fov_changed: 
                    self.setResoultion()

                imgui.tree_pop()

            if imgui.tree_node("Progress"):
                changed, self.var.progress = imgui.checkbox("Progress", self.var.progress)
                if changed:
                    self.progress_image = None # reset image

                imgui.same_line()
                imgui.text(f"t: {self.var.progress_t}")
                imgui.tree_pop()

            if imgui.tree_node("FLIP"):
                self.var.eval_flip = False
                if imgui.button("Evaluate flip"):
                    self.var.eval_flip = True

                changed, self.var.show_flip = imgui.checkbox("Show flip", self.var.show_flip)
                name_list = ["Flip", "GT", "Test"]
                for i, name in enumerate(name_list):
                    if imgui.radio_button(name, self.var.show_flip_type == i):
                        self.var.show_flip_type = i
                    imgui.same_line()
                imgui.new_line()
                imgui.tree_pop()
                
            if imgui.tree_node("Animation"):
                changed, self.var.animate_camera = imgui.checkbox("Camera", self.var.animate_camera)
                changed, self.var.time = imgui.slider_int("Time", self.var.time, 0, self.scene.getMaxTime())
                if changed:
                    self.scene.setTime(self.var.time)
                    if self.var.animate_camera:
                        self.origin, self.angle = self.scene.getCamera(self.var.time)
                    self.view_change = True
                imgui.tree_pop()

            for i, integrator in enumerate(self.integrator_list):
                if imgui.radio_button(integrator.name(), self.var.selected_radio == i):
                    self.var.selected_radio = i
                    self.progress_image = None
                    self.change_integrator(self.var.selected_radio)

                imgui.same_line()
            imgui.new_line()

            self.integrator.gui()
            
            if imgui.button("Random"):
                min, scale = self.scene.scene_bbox_info()
                min = np.array(min)
                scale = np.array(scale)
                self.origin = np.random.random(3) * scale + min
                self.angle = np.random.random(2) * np.array([2*np.pi, np.pi/4])+ np.array([-np.pi, 3*np.pi/8])
                self.view_change = True
            self.var.save = False
            if imgui.button("Save image"):
                self.var.save = True

            self.var.snapshot_output = False
            if imgui.button("Snapshot"):
                self.var.snapshot_output = True

        imgui.end()

        imgui.render()
        gui.render(imgui.get_draw_data())

    @measure_execution_time
    def image_frame(self, image, linear_input = True, float_input = True):
        '''image in tensor (H, W, 3) or (H, W, 4) linear srgb uint8 float'''
        if image.shape[2] == 4:
            image = image[:, :, :3]

        if linear_input:
            image = linear_to_srgb(image)
        if float_input:
            image = float_to_uint8(image)

        self.render_image = image

        w,h = self.var.windowWidth , self.var.windowHeight 
        surface = pygame.surfarray.make_surface(to_np(image).swapaxes(0,1))
        surface = pygame.transform.scale(surface, (w, h))
        pygame.draw.circle(surface, (0,0,255), (w//2, h//2), 3)
        textData = pygame.image.tostring(surface, "RGBA", True)
        gl.glWindowPos2i(0, 0)
        gl.glDrawPixels(w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, textData)
    
    def get_key(self, key):
        speed = self.var.MOVE_SPEED
        previous = np.array(self.origin)
        if key[pygame.K_w]:
            self.origin += self.getDirection() * speed
        if key[pygame.K_s]:
            self.origin -= self.getDirection() * speed

        if key[pygame.K_a]:
            self.origin -= np.cross(self.getDirection(), render_system.Y) * speed
        if key[pygame.K_d]:
            self.origin += np.cross(self.getDirection(), render_system.Y) * speed

        if key[pygame.K_e]:
            self.origin += render_system.Y * speed
        if key[pygame.K_q]:
            self.origin -= render_system.Y * speed

        if key[pygame.K_r]:
            self.origin, self.angle = self.scene.get_default_camera()

        if np.any(previous != self.origin):
            self.view_change = True
    
    def mouse_callback(self, event):
        gui = imgui.get_io()
        if gui.want_capture_mouse:
            return
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.draging = True
                self.pre_pos = np.array(event.pos)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:            
                self.draging = False

        elif event.type == pygame.MOUSEMOTION:
            if self.draging:
                self.view_change = True
                pos = np.array(event.pos)
                offset = self.pre_pos - pos
                offset[1] = -offset[1]
                self.angle += offset * self.var.VIEW_SPEED
                self.pre_pos = pos
            self.angle[1] = np.clip(self.angle[1], 1e-5, np.pi - 1e-5)

    def render_gt(self, total_spp = 512) -> np.ndarray:
        if self.var.sensor_id == -1:
            spp = 32
            for i, integrator in enumerate(self.integrator_list):
                if integrator.name() == "Path":
                    break
            
            self.change_integrator(i)

            progress_image = None
            for t in range(total_spp // spp):
                result = self.render(spp)
                image = self.integrator.postprocess(result, {})
                if progress_image is None:
                    progress_image = torch.zeros_like(image).cuda()
                progress_image = (progress_image * t + image) / (t + 1)

            self.change_integrator(self.var.selected_radio)
            return to_np(progress_image)
        else:
            # load image from dataset
            info = self.dataset.info[self.var.sensor_id]
            img, *_ = self.dataset.load_id(info)
            return cv2.resize(img, (self.width, self.height))

    @measure_execution_time 
    def postprocess(self, images, sys_info):
        ''' 
        - progress
        - save image
        '''
        image = self.integrator.postprocess(images, sys_info)
        
        if self.var.progress:
            if self.progress_image is None or self.progress_image.shape != image.shape:
                self.progress_image = image
                self.var.progress_t = 0
            self.var.progress_t += 1
            self.progress_image = (self.progress_image * (self.var.progress_t - 1) + image) / self.var.progress_t
            image = self.progress_image

        if self.var.save:
            output = self.integrator.save_image(images)
            if output["img"] is None:
                output["img"] = to_np(image)
            self.dataset.write_outputs(output, sys_info)
        
        return image
    
    def init_main(self):
        # Initialize Pygame
        pygame.init()
        screen = pygame.display.set_mode((self.var.windowWidth, self.var.windowHeight), pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
        pygame.display.set_caption(self.var.windowName)

        # Initialize imgui
        imgui.create_context()
        self.gui = PygameRenderer()
        io = imgui.get_io()
        io.fonts.add_font_default()
        io.display_size = (self.var.windowWidth, self.var.windowHeight)
        self.sample_guard()
        # Main loop
        self.running = True

        self.app_live = 0
        self.already_init = True

    def frame_input(self):
        # input
        if self.already_init == False:
            raise RuntimeError("not init")
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
            elif  event.type == pygame.VIDEORESIZE:
                self.var.windowWidth, self.var.windowHeight = event.w, event.h
                self.setResoultion()

            self.mouse_callback(event)
            self.gui.process_event(event)
        self.get_key(pygame.key.get_pressed())
        self.gui.process_inputs()

    def frame_main(self):
        '''called each frame'''
        if self.view_change:
            self.progress_image = None
        self.view_change = False

        self.app_live += 1

        # render image	
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        image = render_system.BLACK
        scene_min, scene_scale = self.scene_bbox_info()

        sys_info = {
            "scene_scale": scene_scale,
            "scene_min": scene_min,
            "origin": self.origin,
            "direction": self.getDirection(),
            "device": self.device,
            "fov": self.var.fov,
            "to_world": self.to_world,
            "scene": self.scene
        }

        if self.var.stop == False and self.var.show_flip == False:
            images = self.render(self.var.SPP)
            image = self.postprocess(images, sys_info)

            self.image_frame(image)

            if self.var.eval_flip:
                self.flip.evaluate(self.render_gt(), to_np(image))
                self.var.show_flip = True

        if self.var.show_flip:
            if self.var.show_flip_type == 1:
                image = self.flip.getGT()
            elif self.var.show_flip_type == 2:
                image = self.flip.getTest()
            else:
                image = self.flip.getErrorMap()
            
            self.image_frame(image, linear_input=False)

        if self.var.snapshot_output and self.render_image is not None:
            self.snapshot.write_images(to_np(self.render_image), sys_info)

        self.gui_frame(self.gui)

        pygame.display.flip()

    def quit_main(self):
        # Quit Pygame
        pygame.quit()
        self.already_init = False

    def main(self):
        try:
            self.init_main()
            while self.running:
                self.frame_input()
                self.frame_main()
        finally:
            self.quit_main()

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
from viewer.my_mi import *
from viewer.dataset import *
from viewer.metric import *
from viewer.my_gl import *
from viewer.gui import Variable, radio_button


class render_system:
    Y = np.array([0,1,0]).astype(float)
    BLACK = torch.zeros((1,1,3)).cuda()
    def __init__(self, scene: mitsuba_scene, dataset:Dataset, snapshot:Dataset, origin = None, angle = None, var:Variable = None, integrators:List[Integrator] = None) -> None:
        self.var = var if var is not None else Variable()
        self.clock = pygame.time.Clock()

        self.scene = scene
        self.origin, self.angle = self.scene.get_default_camera()
        if origin is not None:
            self.origin = np.array(origin).astype(float)
        if angle is not None:
            self.angle = np.array(angle).astype(float)

        self.to_world = mi.Transform4f.look_at(self.origin, self.getTarget(), render_system.Y)

        self.dataset = dataset
        self.snapshot = snapshot

        self.width = 2**self.var.sensor.resoultion
        self.height = self.width
        self.draging = False
        self.progress_image = None

        self.integrator_list : List[Integrator] = []

        if integrators is not None:
            self.integrator_list += integrators

        self.change_integrator(0)

        # reset progress image if view change
        self.view_change = False
        
        self.app_live = 0
        self.waitingTimes = 0

        self.flip = flipError()  

        self.already_init = False

    def change_integrator(self, id):
        assert(id < len(self.integrator_list) and id >= 0)

        self.integrator : Integrator = self.integrator_list[id]
        self.integrator.update(self.var.sensor.depth)
        self.integrator.resize(self.width, self.height, self.var.sensor.fov)

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
            self.to_world = mi.ScalarTransform4f.look_at(self.origin, self.getTarget(), render_system.Y)
            self.integrator.setCameraPose(self.to_world)
            return self.integrator.render(SPP)

    def setResoultion(self):
        '''call when resoultion or window size change'''
        ratio = self.var.window.windowHeight / self.var.window.windowWidth if self.var.window.auto_resize else 1.0
        self.width = min(2 ** self.var.sensor.resoultion, self.var.window.windowWidth)
        self.height = int(self.width * ratio)
        self.integrator.resize(self.width, self.height, self.var.sensor.fov)
        self.sample_guard()
        self.view_change = True

        if withPyCuda:
            self.cuda_gl_texture.unregister()
            delete_gl_texture(self.texture_id)
            self.texture_id , self.cuda_gl_texture= create_map_texture(self.width, self.height)

    def sample_guard(self):
        '''prevent low FPS'''
        if (self.var.sensor.SPP * self.height * self.width) > self.var.sensor.limit:
            self.var.stop = True

    def gui_frame(self, gui):
        imgui.new_frame()
        
        self.clock.tick()

        self.var.once.reset()

        is_expand, show_custom_window = imgui.begin("Custom window", True)
        if is_expand:
            # information
            fps = self.clock.get_fps()
            if fps < 3 and self.app_live > 30 and self.var.fps_guard:
                self.waitingTimes += 1
                if self.waitingTimes > 5:
                    self.var.stop = True
            else:
                self.waitingTimes = 0

            imgui.text(f"FPS: {fps}")
            imgui.text(f"Window width:{self.var.window.windowWidth} height:{self.var.window.windowHeight}")
            imgui.text(f"Image width:{self.width} height:{self.height}")
            imgui.text(f"Position: {np.round(self.origin, 3)}")
            imgui.text(f"Direction: {np.round(self.angle, 3)}")

            changed, self.var.stop = imgui.checkbox("Stop", self.var.stop)
            if changed: self.sample_guard()

            if imgui.tree_node("Sensor"):
                resoultion_changed, self.var.window.auto_resize = imgui.checkbox("Fit to Window", self.var.window.auto_resize)

                changed, self.var.sensor.resoultion = imgui.slider_int("resoultion", self.var.sensor.resoultion, 5, 11)
                resoultion_changed = resoultion_changed or changed
                
                _, self.var.sensor.SPP = imgui.slider_int("SPP", self.var.sensor.SPP, 1, 256)

                depth_change, self.var.sensor.depth = imgui.slider_int("depth", self.var.sensor.depth, 1, 10)

                fov_changed, self.var.sensor.fov = imgui.slider_int("fov", self.var.sensor.fov, 10, 90)

                if self.dataset is not None:
                    sensor_changed, self.var.selected_dataset_sensor = imgui.slider_int("Sensor(dataset)", self.var.selected_dataset_sensor, -1, self.dataset.n_sensor-1)
                    if sensor_changed:
                        fov_changed = True
                        if self.var.selected_dataset_sensor != -1:
                            info = self.dataset.info[self.var.selected_dataset_sensor]
                            # TODO
                            self.var.sensor.fov, to_world, self.origin = info['fov'], info['to_world'], info['origin']
                            self.origin, self.angle = matrix2camera(to_world)


                if depth_change:
                    self.integrator.update(self.var.sensor.depth)
                    self.view_change = True

                if resoultion_changed or fov_changed: 
                    self.setResoultion()

                changed, self.var.selected_scene_sensor = imgui.combo("Sensor(scene)", self.var.selected_scene_sensor, self.scene.cameras_id)
                if changed:
                    self.scene.setCamera(self.var.selected_scene_sensor)
                    self.origin, self.angle = self.scene.getCamera(self.var.animation.time)
                    
                imgui.tree_pop()

            if imgui.tree_node("Progress"):
                changed = self.var.progress.gui()
                if changed:
                    self.progress_image = None # reset image
                imgui.tree_pop()

            if imgui.tree_node("FLIP"):
                if imgui.button("Evaluate flip"):
                    self.var.once.eval_flip = True
                self.var.flip.gui()
                imgui.tree_pop()
                
            if imgui.tree_node("Animation"):
                changed = self.var.animation.gui(self.scene.getMaxTime())
                if changed:
                    self.scene.setTime(self.var.animation.time)
                    if self.var.animation.animate_camera:
                        self.origin, self.angle = self.scene.getCamera(self.var.animation.time)
                    self.view_change = True
                imgui.tree_pop()
            
            changed, self.var.selected_integrator = radio_button([i.name() for i in self.integrator_list], self.var.selected_integrator)
            if changed:
                self.progress_image = None
                self.change_integrator(self.var.selected_integrator)

            self.integrator.gui()
            
            if imgui.button("Save image"):
                self.var.once.save_output = True

            if imgui.button("Snapshot"):
                self.var.once.snapshot_output = True

        imgui.end()

        imgui.render()
        gui.render(imgui.get_draw_data())

    @measure_execution_time
    def image_frame(self, image: torch.Tensor, linear_input = True, float_input = True, cudaTensor = True):
        '''image in tensor (H, W, 3) or (H, W, 4) linear srgb uint8 float'''
        if image.shape[2] == 4:
            image = image[:, :, :3]

        if linear_input:
            image = linear_to_srgb(image)
        if float_input:
            image = float_to_uint8(image)

        self.render_image = image

        # add a center point
        center = image.shape[0]//2, image.shape[1]//2
        image[center[0] -2 : center[0] + 2, center[1]-2:center[1] + 2] = torch.tensor([0,0,255]) # blue color

        if withPyCuda and cudaTensor:
            copy_tensor_to_texture(image, self.texture_id, self.cuda_gl_texture)
            render_texture(self.texture_id)
        else:
            render_cpu_array(to_np(image), self.var.window.windowWidth , self.var.window.windowHeight)
    
    def get_key(self, key):
        speed = self.var.speed.move
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
            self.origin, self.angle = self.scene.getCamera(self.var.animation.time)

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
                self.angle += offset * self.var.speed.view
                self.pre_pos = pos
            self.angle[1] = np.clip(self.angle[1], 1e-5, np.pi - 1e-5)

    def render_gt(self, total_spp = 512) -> np.ndarray:
        if self.var.selected_dataset_sensor == -1:
            spp = 32
            for i, integrator in enumerate(self.integrator_list):
                if integrator.name() == "Path":
                    break
            else:
                print("Path integrator not found!")
                return np.zeros((self.width, self.height, 3))
            
            self.change_integrator(i)

            progress_image = None
            for t in range(total_spp // spp):
                result = self.render(spp)
                image = self.integrator.postprocess(result, {})
                if progress_image is None:
                    progress_image = torch.zeros_like(image).cuda()
                progress_image = (progress_image * t + image) / (t + 1)

            self.change_integrator(self.var.selected_integrator)
            return to_np(progress_image)
        else:
            # load image from dataset
            info = self.dataset.info[self.var.selected_dataset_sensor]
            img, *_ = self.dataset.load_id(info)
            return cv2.resize(img, (self.width, self.height))

    @measure_execution_time 
    def postprocess(self, images, sys_info):
        ''' 
        - progress
        - save image
        '''
        image = self.integrator.postprocess(images, sys_info)
        
        if self.var.progress.enable:
            if self.progress_image is None or self.progress_image.shape != image.shape:
                self.progress_image = torch.zeros_like(image, device="cuda")
                self.var.progress.t = 0
            self.var.progress.t += 1
            self.progress_image = (self.progress_image * (self.var.progress.t - 1) + image) / self.var.progress.t
            image = self.progress_image

        if self.var.once.save_output:
            output = self.integrator.save_image(images)
            if output["img"] is None:
                output["img"] = to_np(image)
            self.dataset.write_outputs(output, sys_info)
        
        return image
    
    def init_main(self):
        global withPyCuda
        # Initialize Pygame
        pygame.init()
        screen = pygame.display.set_mode((self.var.window.windowWidth, self.var.window.windowHeight), pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
        pygame.display.set_caption(self.var.window.windowName)

        
        try:
            if withPyCuda:
                import pycuda.gl.autoinit
                gl.glEnable(gl.GL_TEXTURE_2D)
                self.texture_id , self.cuda_gl_texture = create_map_texture(self.width, self.height)
        except:
            withPyCuda = False
        print("With PyCuda =", withPyCuda)

        # Initialize imgui
        imgui.create_context()
        self.gui = PygameRenderer()
        io = imgui.get_io()
        io.fonts.add_font_default()
        io.display_size = (self.var.window.windowWidth, self.var.window.windowHeight)
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
                self.var.window.windowWidth, self.var.window.windowHeight = event.w, event.h
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
            "fov": self.var.sensor.fov,
            "to_world": self.to_world,
            "scene": self.scene
        }

        if self.var.stop == False and self.var.flip.display == False:
            images = self.render(self.var.sensor.SPP)
            image = self.postprocess(images, sys_info)

            if self.var.once.eval_flip:
                self.flip.evaluate(self.render_gt(), to_np(image[:,:,:3]))
                self.var.flip.display = True

            self.image_frame(image)

            

        if self.var.flip.display:
            if self.var.flip.selected_type == 1:
                image = self.flip.getGT()
            elif self.var.flip.selected_type == 2:
                image = self.flip.getTest()
            else:
                image = self.flip.getErrorMap()
            
            self.image_frame(torch.tensor(image), linear_input=False, cudaTensor=False)

        if self.var.once.snapshot_output and self.render_image is not None:
            self.snapshot.write_images(to_np(self.render_image), sys_info)

        self.gui_frame(self.gui)
        pygame.display.flip()

    def quit_main(self):
        # Quit Pygame
        if withPyCuda:
            self.cuda_gl_texture.unregister()
            delete_gl_texture(self.texture_id)
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

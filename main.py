import json
import viewer
from viewer.common import *
from viewer.execution_time import *

import mitsuba as mi
import numpy as np

from viewer import my_mi

def miWrapper_QuickStart():
    sceneRoot = "./scene/miScene"

    sceneFile = os.path.join(sceneRoot, "scene.xml")
    animationFile = os.path.join(sceneRoot, "animation.xml")
    scene = my_mi.mitsuba_scene(sceneFile, animationFile)

    n_sensor = len(scene.object.sensors())
    
    # render image at time t
    t = 2500

    # animated object
    scene.setTime(t)

    # activae sensor, if no sensor in scene.xml. this will set activated camera position to (0,0,0)
    sensorIndex = 0
    sensorPose = scene.getCameraMatrix(t, sensorIndex)

    print(f"Number of sensors: {n_sensor}")
    print(f"Number of animations (object + sensor): {len(scene.animation)}")
    print(f"Use camera {sensorIndex} to render image at time {t} ")

    '''use sensor in the scene.xml'''
    if n_sensor > 0:
        sensor = my_mi.mitsiba_sensor(scene.object.sensors()[sensorIndex])

        # apply animation to sensor (if present)
        # this will change sensor parameter directly
        # set pose to time = 0 to reset pose
        sensor.setCameraPose(sensorPose)

        image = mi.render(scene.object, sensor = sensorIndex, spp=256)

    
    '''or create new sensor to redner image'''
    sensor = my_mi.mitsiba_sensor()
    width, height, fov = 512, 512, 45
    sensor.resize(width, height, fov)

    # apply animation to sensor (if present)
    sensor.setCameraPose(sensorPose)

    image = mi.render(scene.object, sensor = sensor.object, spp=256)


    ''' use my_integrator '''
    pathIntegrator = my_mi.path_Integrator()
    pathIntegrator.update(depth = 8)

    images = pathIntegrator.render(scene, sensor, 256)
    image = to_np(pathIntegrator.postprocess(images))

    return image


def mitsuba_interactive_window(path, outputFolder):
    sceneFile = os.path.join(path, "scene.xml")
    animationFile = os.path.join(path, "animation.xml")
    scene = my_mi.mitsuba_scene(sceneFile, animationFile)

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    dataset = viewer.Dataset(os.path.join(outputFolder, "images"), 512)
    sanpshot = viewer.Dataset(os.path.join(outputFolder, "snapshot"))

    integrators = [
        my_mi.path_Integrator(),
        my_mi.path_info_Integrator(),
        my_mi.ptrace_Integrator()
    ]
    
    sys = viewer.render_system(scene, dataset, snapshot = sanpshot, integrators= integrators)
    
    start_time = time.time()
    
    clear_total_execution_time()
    sys.main()
    print_total_execution_time()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Main execution time: {round(execution_time, 6)}")

    return sys

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A script that accepts a path argument")
    parser.add_argument("-p", "--path", type=str, default="./scene/miScene/", help="Path to the directory (default: './scene/miScene/')")
    parser.add_argument("-o", "--output", type=str, default="./output/", help="Output path to the directory (default: './output/')")

    args = parser.parse_args()
    mitsuba_interactive_window(args.path, args.output)

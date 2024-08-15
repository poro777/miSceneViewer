import json
import viewer
from viewer.common import *
from viewer.execution_time import *

import mitsuba as mi
import numpy as np

def mitsuba_interactive_window(path, outputFolder):
    from viewer import my_mi

    sceneFile = os.path.join(path, "scene.xml")
    animationFile = os.path.join(path, "animation.xml")
    scene = my_mi.mitsuba_scene(sceneFile, animationFile)

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    dataset = viewer.Dataset(os.path.join(outputFolder, "images"), 512)
    sanpshot = viewer.Dataset(os.path.join(outputFolder, "snapshot"))

    integrators = [
        my_mi.path_Integrator(scene.object),
        my_mi.path_info_Integrator(scene.object),
        my_mi.ptrace_Integrator(scene.object)
    ]
    
    sys = viewer.render_system(scene, dataset= dataset, integrators= integrators, snapshot = sanpshot)
    
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

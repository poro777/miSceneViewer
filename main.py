import json
import viewer
from viewer.common import *
from viewer.execution_time import *

import mitsuba as mi
import numpy as np

def mitsuba_interactive_window(path, output):
    from viewer import my_mi

    sceneFile = os.path.join(path, "scene.xml")
    animationFile = os.path.join(path, "animation.xml")
    scene = my_mi.mitsuba_scene(sceneFile, animationFile)

    # TODO
    
    dataset = viewer.Dataset(os.path.join(output, "images"), 512)
    sanpshot = viewer.Dataset(os.path.join(output, "snapshot"))

    integrators = [
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
    outputFolder = "./output/"
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    mitsuba_interactive_window("./scene/miScene/", outputFolder)

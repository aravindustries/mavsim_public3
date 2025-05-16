import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))
# use QuitListener for Linux or PC <- doesn't work on Mac
#from tools.quit_listener import QuitListener
import numpy as np
import parameters.simulation_parameters as SIM
from message_types.msg_delta import MsgDelta
from models.mav_dynamics import MavDynamics
from viewers.view_manager import ViewManager
import time

#quitter = QuitListener()
    
# initialize elements of the architecture
mav = MavDynamics(SIM.ts_simulation)
delta = MsgDelta()
viewers = ViewManager(data=True,
                      video=False, animation=True, video_name='chap3.mp4')

# initialize the simulation time
sim_time = SIM.start_time
end_time = 60

# main simulation loop
print("Press 'Esc' to exit...")
while sim_time < end_time:
    # ------- vary forces and moments to check dynamics -------------
    fx = 0  # 10
    fy = 0  # 10
    fz =  10
    Mx = 0
    My = 0  # 0.1
    Mz = 0  # 0.1
    forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T

    # ------- physical system -------------
    mav.update(forces_moments)  # propagate the MAV dynamics

    # ------- update viewers -------------
    viewers.update(
        sim_time,
        true_state=mav.true_state,  # true states
    )

    # ------- increment time -------------
    sim_time += SIM.ts_simulation
    time.sleep(0.002) # slow down the simulation for visualization

    # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break

# Save an Image of the Plot
viewers.close(dataplot_name="ch3_data_plot")
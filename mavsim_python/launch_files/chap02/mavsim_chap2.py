import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
import pyqtgraph as pg
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))
# use QuitListener for Linux or PC <- doesn't work on Mac
#from python_tools.quit_listener import QuitListener
import parameters.simulation_parameters as SIM
from message_types.msg_state import MsgState
from viewers.view_manager import ViewManager
# from viewers.mav_viewer import MavViewer
import time

# #quitter = QuitListener()
state = MsgState()
app = pg.QtWidgets.QApplication([])
# mav_viewer = MavViewer(app=app)
viewers = ViewManager(video=False, animation=True, video_name='chap2.mp4')

# initialize the simulation time
sim_time = SIM.start_time
motions_time = 0
time_per_motion = 3
end_time = 20

# main simulation loop
print("Press Esc to exit...")

while sim_time < end_time:
    # -------vary states to check viewer-------------
    if motions_time < time_per_motion:
        state.north += 10*SIM.ts_simulation
    elif motions_time < time_per_motion*2:
        state.east += 10*SIM.ts_simulation
    elif motions_time < time_per_motion*3:
        state.altitude += 10*SIM.ts_simulation
    elif motions_time < time_per_motion*4:
        state.psi += 0.1*SIM.ts_simulation
    elif motions_time < time_per_motion*5:
        state.theta += 0.1*SIM.ts_simulation
    else:
        state.phi += 0.1*SIM.ts_simulation
    # -------update viewer and video-------------
    # mav_viewer.update(state)
    # mav_viewer.process_app()
    viewers.update(
        sim_time,
        true_state=state,  # true states
    )

    # -------increment time-------------
    sim_time += SIM.ts_simulation
    motions_time += SIM.ts_simulation
    if motions_time >= time_per_motion*6:
        motions_time = 0
    time.sleep(0.002) # slow down the simulation for visualization

    # # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break


import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))
# use QuitListener for Linux or PC <- doesn't work on Mac
#from python_tools.quit_listener import QuitListener
import pyqtgraph as pg
import parameters.simulation_parameters as SIM
from viewers.spacecraft_viewer import SpaceCraftViewer
from message_types.msg_state import MsgState

#quitter = QuitListener()
VIDEO = False
if VIDEO is True:
    from viewers.video_writer import VideoWriter
    video = VideoWriter(video_name="chap2_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)
# initialize the visualization
app = pg.QtWidgets.QApplication([])
spacecraft_view = SpaceCraftViewer(app=app)  
# initialize elements of the architecture
state = MsgState()


# initialize the simulation time
sim_time = SIM.start_time
motions_time = 0
time_per_motion = 3
end_time = 20

# main simulation loop
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
    spacecraft_view.update(state)
    spacecraft_view.process_app()

    # -------increment time-------------
    sim_time += SIM.ts_simulation
    motions_time += SIM.ts_simulation
    if motions_time >= time_per_motion*6:
        motions_time = 0

    # -------update video---------------
    if VIDEO is True:
        video.update(sim_time)

    # # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break

if VIDEO is True:
    video.update(sim_time)

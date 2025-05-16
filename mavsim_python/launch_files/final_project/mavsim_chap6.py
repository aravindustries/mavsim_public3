import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))
# use QuitListener for Linux or PC <- doesn't work on Mac
#from tools.quit_listener import QuitListener
import numpy as np
import parameters.simulation_parameters as SIM
from tools.signals import Signals
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
# from controllers.autopilot import Autopilot
#from controllers.autopilot_tecs import Autopilot
from controllers.autopilot import Autopilot
from viewers.view_manager import ViewManager
import time

#quitter = QuitListener()

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
autopilot = Autopilot(SIM.ts_simulation)
viewers = ViewManager(animation=True, 
                      data=True,
                      video=False)

# autopilot commands
from message_types.msg_autopilot import MsgAutopilot
commands = MsgAutopilot()
Va_command = Signals(dc_offset=25.0,
                     amplitude=3.0,
                     start_time=2.0,
                     frequency=0.01)
altitude_command = Signals(dc_offset=100.0,
                           amplitude=20.0,
                           start_time=0.0,
                           frequency=0.02)
course_command = Signals(dc_offset=np.radians(180),
                         amplitude=np.radians(45),
                         start_time=5.0,
                         frequency=0.015)

# initialize the simulation time
sim_time = SIM.start_time
end_time = 100

# Lists for RMSE logging
true_Va_log = []
cmd_Va_log = []
true_alt_log = []
cmd_alt_log = []
true_chi_log = []
cmd_chi_log = []

# main simulation loop
print("Press 'Esc' to exit...")
while sim_time < end_time:

    # -------autopilot commands-------------
    commands.airspeed_command = Va_command.square(sim_time)
    commands.course_command = course_command.square(sim_time)
    commands.altitude_command = altitude_command.square(sim_time)

    # -------autopilot-------------
    estimated_state = mav.true_state  # uses true states in the control
    delta, commanded_state = autopilot.update(commands, estimated_state)

    # -------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # ------- update viewers -------
    viewers.update(
        sim_time,
        true_state=mav.true_state,  # true states
        commanded_state=commanded_state,  # commanded states
        delta=delta, # inputs to MAV
    )

    # Log the states
    true_Va_log.append(mav.true_state.Va)
    cmd_Va_log.append(commanded_state.Va)

    true_alt_log.append(mav.true_state.altitude)
    cmd_alt_log.append(commanded_state.altitude)

    true_chi_log.append(mav.true_state.chi)
    cmd_chi_log.append(commanded_state.chi)

    # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break

    # -------increment time-------------
    sim_time += SIM.ts_simulation
    time.sleep(0.001) # slow down the simulation for visualization

viewers.close(dataplot_name="ch6_data_plot")

def compute_rmse(true_vals, cmd_vals):
    return np.sqrt(np.mean((np.array(true_vals) - np.array(cmd_vals))**2))

Va_rmse = compute_rmse(true_Va_log, cmd_Va_log)
alt_rmse = compute_rmse(true_alt_log, cmd_alt_log)
chi_rmse = compute_rmse(true_chi_log, cmd_chi_log)

print(f"\n--- Simulation RMSE ---")
print(f"Airspeed RMSE: {Va_rmse:.2f} m/s")
print(f"Altitude RMSE: {alt_rmse:.2f} m")
print(f"Course RMSE: {np.degrees(chi_rmse):.2f} deg")

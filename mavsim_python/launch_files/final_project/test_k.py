"""
mavsim_python
    - Chapter 6 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/5/2019 - RWB
        2/24/2020 - RWB
        1/5/2023 - David L. Christiansen
        7/13/2023 - RWB
"""
import csv
import os, sys
import argparse
import warnings
warnings.simplefilter('ignore')
# insert parent directory at beginning of python search path
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
from viewers.view_manager import ViewManager
import time
import parameters.control_parameters as AP
# from tools.transfer_function import TransferFunction
from tools.wrap import wrap
from controllers.pi_control import PIControl
from controllers.pd_control_with_rate import PDControlWithRate
from controllers.tf_control import TFControl
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
from tools.transfer_function import TransferFunction

parser = argparse.ArgumentParser()
parser.add_argument("--cki", type=float, default=AP.course_ki)
parser.add_argument("--ckp", type=float, default=AP.course_kp)
parser.add_argument("--rkp", type=float, default=AP.roll_kp)
parser.add_argument("--rkd", type=float, default=AP.roll_kd)
parser.add_argument("--yk", type=float, default=AP.yaw_damper_kr)
parser.add_argument("--yw", type=float, default=AP.yaw_damper_p_wo)
parser.add_argument("--alki", type=float, default=AP.altitude_ki)
parser.add_argument("--alkp", type=float, default=AP.altitude_kp)
parser.add_argument("--pkp", type=float, default=AP.pitch_kp)
parser.add_argument("--pkd", type=float, default=AP.pitch_kd)
parser.add_argument("--askp", type=float, default=AP.airspeed_throttle_kp)
parser.add_argument("--aski", type=float, default=AP.airspeed_throttle_ki)
parser.add_argument("--outfile", type=str, default="launch_files/chap06/rmse_results21.csv")
args = parser.parse_args()
#print(args)

def saturate(self, input, low_limit, up_limit):
    if input <= low_limit:
        output = low_limit
    elif input >= up_limit:
        output = up_limit
    else:
        output = input
    return output

class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral-directional controllers
        self.roll_from_aileron = PDControlWithRate(
                        kp=args.rkp,
                        kd=args.rkd,
                        limit=np.radians(45))
        self.course_from_roll = PIControl(
                        kp=args.ckp,
                        ki=args.cki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.yaw_damper = TransferFunction(
                        num=np.array([[args.yk, 0]]),
                        den=np.array([[1, args.yw]]),
                        Ts=ts_control)
        self.pitch_from_elevator = PDControlWithRate(
                        kp=args.pkp,
                        kd=args.pkd,
                        limit=np.radians(45))
        self.altitude_from_pitch = PIControl(
                        kp=args.alkp,
                        ki=args.alki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.airspeed_from_throttle = PIControl(
                        kp=args.askp,
                        ki=args.aski,
                        Ts=ts_control,
                        limit=1.0)
        self.commanded_state = MsgState()

    def update(self, cmd, state):
        # lateral autopilot
        chi_c = wrap(cmd.course_command, state.chi)
        
        # Course hold
        phi_c = self.course_from_roll.update(chi_c, state.chi)
        
        # Roll hold
        delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p)
        
        # Yaw damper
        delta_r = self.yaw_damper.update(state.r)
        
        # longitudinal autopilot
        # Airspeed hold with throttle
        delta_t = self.airspeed_from_throttle.update(cmd.airspeed_command, state.Va)
        
        # Altitude hold
        theta_c = self.altitude_from_pitch.update(cmd.altitude_command, state.altitude)
        
        # Pitch hold
        delta_e = self.pitch_from_elevator.update(theta_c, state.theta, state.q)
        
        # construct control outputs and commanded states
        delta = MsgDelta(elevator=delta_e,
                        aileron=delta_a,
                        rudder=delta_r,
                        throttle=delta_t)
        
        # Update commanded state to match format in LQR version
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = chi_c
        
        return delta, self.commanded_state


#quitter = QuitListener()

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
autopilot = Autopilot(SIM.ts_simulation)
viewers = ViewManager(animation=False, 
                      data=False,
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
end_time = 140

# Lists for RMSE logging
true_Va_log = []
cmd_Va_log = []
true_alt_log = []
cmd_alt_log = []
true_chi_log = []
cmd_chi_log = []

# main simulation loop
#print("Press 'Esc' to exit...")
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
    #time.sleep(0.001) # slow down the simulation for visualization

viewers.close(dataplot_name="ch6_data_plot")

def compute_rmse(true_vals, cmd_vals):
    return np.sqrt(np.mean((np.array(true_vals) - np.array(cmd_vals))**2))

def compute_tv(values):
    values = np.array(values)
    return np.sum(np.abs(np.diff(values)))

Va_rmse = compute_rmse(true_Va_log, cmd_Va_log)
alt_rmse = compute_rmse(true_alt_log, cmd_alt_log)
chi_rmse = compute_rmse(true_chi_log, cmd_chi_log)

Va_tv = compute_tv(true_Va_log)
alt_tv = compute_tv(true_alt_log)
chi_tv = compute_tv(true_chi_log)

#print(f"\n--- Simulation RMSE ---")
#print(f"Airspeed RMSE: {Va_rmse:.2f} m/s")
#print(f"Altitude RMSE: {alt_rmse:.2f} m")
#print(f"Course RMSE: {np.degrees(chi_rmse):.2f} deg")

header = ["cki", "ckp", "rkp", "rkd", "yk", "yw",
          "alki", "alkp", "pkp", "pkd", "askp", "aski",
          "Va_RMSE", "Alt_RMSE", "Chi_RMSE_deg",
          "Va_TV", "Alt_TV", "Chi_TV_deg"]
row = [args.cki, args.ckp, args.rkp, args.rkd, args.yk, args.yw,
       args.alki, args.alkp, args.pkp, args.pkd, args.askp, args.aski,
       round(Va_rmse, 4), round(alt_rmse, 4), round(np.degrees(chi_rmse), 4),
       round(Va_tv, 4), round(alt_tv, 4), round(np.degrees(chi_tv), 4)]

# Ensure output directory exists
os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

file_exists = os.path.isfile(args.outfile)
with open(args.outfile, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(header)
    writer.writerow(row)
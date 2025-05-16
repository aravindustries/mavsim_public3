import os, sys
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))
import numpy as np
import parameters.simulation_parameters as SIM
from models.mav_dynamics_control import MavDynamics
from message_types.msg_delta import MsgDelta
from models.wind_simulation import WindSimulation

wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
delta = MsgDelta()
delta.elevator = -0.2
delta.aileron = 0.0
delta.rudder = 0.005
delta.throttle = 0.5

T_p, Q_p = mav._motor_thrust_torque(mav._Va, delta.throttle)
print("Propeller Forces and Torque", "\n")
print("T_p: " , T_p)
print("Q_p: " , Q_p, "\n\n")

forcesAndMoments = mav._forces_moments(delta)
print("Forces and Moments : Case 1", "\n")
print("fx: " , forcesAndMoments.item(0))
print("fy: " , forcesAndMoments.item(1))
print("fz: " , forcesAndMoments.item(2))
print("Mx: " , forcesAndMoments.item(3))
print("My: " , forcesAndMoments.item(4))
print("Mz: " , forcesAndMoments.item(5) , "\n\n")

x_dot = mav._f(mav._state, forcesAndMoments)
print("State Derivatives : Case 1", "\n")
print("north_dot: ", x_dot.item(0))
print("east_dot: ", x_dot.item(1))
print("down_dot: ", x_dot.item(2))
print("   u_dot: ", x_dot.item(3))
print("   v_dot: ", x_dot.item(4))
print("   w_dot: ", x_dot.item(5))
print("  e0_dot: ", x_dot.item(6))
print("  e1_dot: ", x_dot.item(7))
print("  e2_dot: ", x_dot.item(8))
print("  e3_dot: ", x_dot.item(9))
print("   p_dot: ", x_dot.item(10))
print("   q_dot: ", x_dot.item(11))
print("    r_dt: ", x_dot.item(12) , "\n\n\n")

##### Case 1 ######

delta.elevator = -0.15705144
delta.aileron = 0.01788999
delta.rudder = 0.01084654
delta.throttle = 1.

mav._state = np.array([[ 6.19506532e+01],
 [ 2.22940203e+01],
 [-1.10837551e+02],
 [ 2.73465947e+01],
 [ 6.19628233e-01],
 [ 1.42257772e+00],
 [ 9.38688796e-01],
 [ 2.47421558e-01],
 [ 6.56821468e-02],
 [ 2.30936730e-01],
 [ 4.98772167e-03],
 [ 1.68736005e-01],
 [ 1.71797313e-01]])

T_p, Q_p = mav._motor_thrust_torque(mav._Va, delta.throttle)
print("Propeller Forces and Torque", "\n")
print("T_p: " , T_p)
print("Q_p: " , Q_p, "\n\n")

forcesAndMoments = mav._forces_moments(delta)
print("Forces and Moments : Case 2" , "\n")
print("fx: " , forcesAndMoments.item(0))
print("fy: " , forcesAndMoments.item(1))
print("fz: " , forcesAndMoments.item(2))
print("Mx: " , forcesAndMoments.item(3))
print("My: " , forcesAndMoments.item(4))
print("Mz: " , forcesAndMoments.item(5) , "\n\n")

x_dot = mav._f(mav._state, forcesAndMoments)
print("State Derivatives : Case 2", "\n")
print("north_dot: ", x_dot.item(0))
print("east_dot: ", x_dot.item(1))
print("down_dot: ", x_dot.item(2))
print("   u_dot: ", x_dot.item(3))
print("   v_dot: ", x_dot.item(4))
print("   w_dot: ", x_dot.item(5))
print("  e0_dot: ", x_dot.item(6))
print("  e1_dot: ", x_dot.item(7))
print("  e2_dot: ", x_dot.item(8))
print("  e3_dot: ", x_dot.item(9))
print("   p_dot: ", x_dot.item(10))
print("   q_dot: ", x_dot.item(11))
print("    r_dt: ", x_dot.item(12) , "\n\n\n")


current_wind = np.array([[ 0.        ],
 [ 0.        ],
 [ 0.        ],
 [-0.00165177],
 [-0.00475441],
 [-0.01717199]])


mav._update_velocity_data(current_wind)
print("Wind Update" , "\n")
print("Va: ", mav._Va)
print("alpha: ", mav._alpha)
print("beta: ", mav._beta , "\n\n")

T_p, Q_p = mav._motor_thrust_torque(mav._Va, delta.throttle)
print("Propeller Forces and Torque", "\n")
print("T_p: " , T_p)
print("Q_p: " , Q_p, "\n\n")

forcesAndMoments = mav._forces_moments(delta)
print("Forces and Moments : Case w/Wind" , "\n")
print("fx: " , forcesAndMoments.item(0))
print("fy: " , forcesAndMoments.item(1))
print("fz: " , forcesAndMoments.item(2))
print("Mx: " , forcesAndMoments.item(3))
print("My: " , forcesAndMoments.item(4))
print("Mz: " , forcesAndMoments.item(5) , "\n\n")

x_dot = mav._f(mav._state, forcesAndMoments)
print("State Derivatives : Case w/Wind", "\n")
print("north_dot: ", x_dot.item(0))
print("east_dot: ", x_dot.item(1))
print("down_dot: ", x_dot.item(2))
print("   u_dot: ", x_dot.item(3))
print("   v_dot: ", x_dot.item(4))
print("   w_dot: ", x_dot.item(5))
print("  e0_dot: ", x_dot.item(6))
print("  e1_dot: ", x_dot.item(7))
print("  e2_dot: ", x_dot.item(8))
print("  e3_dot: ", x_dot.item(9))
print("   p_dot: ", x_dot.item(10))
print("   q_dot: ", x_dot.item(11))
print("    r_dt: ", x_dot.item(12) , "\n\n\n")

import numpy as np
from scipy.optimize import minimize
from tools.rotations import euler_to_quaternion
from message_types.msg_delta import MsgDelta
import time

def compute_trim(mav, Va, gamma):
    # define initial state and input

    ##### TODO #####
    # set the initial conditions of the optimization
    alpha = 0.0 # angle of attack, change if needed
    theta = alpha + gamma
    e0 = euler_to_quaternion(0., theta, 0.)
    state0 = np.array([[0.],  # pn
                   [0.],  # pe
                   [0.],  # pd
                   [Va],  # u
                   [0.], # v
                   [0.], # w
                   [e0[0,0]],  # e0
                   [e0[1,0]],  # e1
                   [e0[2,0]],  # e2
                   [e0[3,0]],  # e3
                   [0.], # p
                   [0.], # q
                   [0.]  # r
                   ])
    delta0 = np.array([[0],  # elevator
                       [0],  # aileron
                       [0],  # rudder
                       [0]]) # throttle
    x0 = np.concatenate((state0, delta0), axis=0)
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7],  # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9],  # e3=0
                                x[10],  # p=0  - angular rates should all be zero
                                x[11],  # q=0
                                x[12],  # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs

    res = minimize(trim_objective_fun, x0.flatten(), method='SLSQP', args=(mav, Va, gamma),
                   constraints=cons, 
                   options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = MsgDelta(elevator=res.x.item(13),
                          aileron=res.x.item(14),
                          rudder=res.x.item(15),
                          throttle=res.x.item(16))
    trim_input.print()
    print('trim_state=', trim_state.T)
    return trim_state, trim_input


def trim_objective_fun(x, mav, Va, gamma):
    # objective function to be minimized
    ##### TODO #####
    state = state = x[:13]
    delta = MsgDelta(elevator=x.item(13),
                      aileron=x.item(14),
                      rudder=x.item(15),
                      throttle=x.item(16))
    desired_trim_state_dot = np.array([
                            [0.],  # pn
                            [0.],  # pe
                            [-Va * np.sin(gamma)],  # pd
                            [0.],  # u
                            [0.], # v
                            [0.], # w
                            [0.],  # e0
                            [1],  # e1
                            [0.],  # e2
                            [0.],  # e3
                            [0.], # p
                            [0.], # q
                            [0.]  # r
                            ])


    # set state and input of the aircraft
    mav._state = state
    mav._update_velocity_data()

    forces_moments = mav._forces_moments(delta) # get forces and moments
    f = mav._f(state, forces_moments) # get state derivatives

    J = np.linalg.norm(desired_trim_state_dot[2:13] - f[2:13])**2

    return J

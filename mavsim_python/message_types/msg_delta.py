import numpy as np


class MsgDelta:
    def __init__(self,
                 elevator=0.0,
                 aileron=0.0,
                 rudder=0.0,
                 throttle=0.5,
                 azimuth_cmd=0.0,
                 elevation_cmd=0.0):
        self.elevator = elevator  # elevator command
        self.aileron = aileron  # aileron command
        self.rudder = rudder  # rudder command
        self.throttle = throttle  # throttle command
        self.gimbal_az = azimuth_cmd  # azimuth command for gimbal
        self.gimbal_el = elevation_cmd  # elevation command for gimbal

    def to_array(self):
        return np.array([[self.elevator],
                         [self.aileron],
                         [self.rudder],
                         [self.throttle],
                         [self.gimbal_az],
                         [self.gimbal_el],
                         ])

    def from_array(self, u):
        self.elevator = u.item(0)
        self.aileron = u.item(1)
        self.rudder = u.item(2)
        self.throttle = u.item(3)
        self.gimbal_az = u.item(4)
        self.gimbal_el = u.item(5)

    def print(self):
        print('elevator=', self.elevator,
              'aileron=', self.aileron,
              'rudder=', self.rudder,
              'throttle=', self.throttle,
              'azimuth_cmd=', self.gimbal_az,
              'elevation_cmd=', self.gimbal_el)


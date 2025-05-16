

class MsgAutopilot:
    def __init__(self):
        self.airspeed_command = 0.0  # commanded airspeed m/s
        self.course_command = 0.0  # commanded course angle in rad
        self.altitude_command = 0.0  # commanded altitude in m
        self.phi_feedforward = 0.0  # feedforward command for roll angle

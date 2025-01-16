"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        1/13/2021 - TWM
        7/13/2023 - RWB
        1/16/2024 - RWB
"""
import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import euler_to_rotation
from tools.drawing import rotate_points, translate_points, points_to_mesh


class DrawMav:
    def __init__(self, state, window, scale=10):
        """
        Draw the MAV.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.north  # north position
            state.east  # east position
            state.altitude   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        self.unit_length = scale
        sc_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        # convert North-East Down to East-North-Up for rendering
        self.R_ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        # get points that define the non-rotated, non-translated spacecraft and the mesh colors
        self.sc_points, self.sc_index, self.sc_meshColors = self.get_sc_points()
        self.sc_body = self.add_object(
            self.sc_points,
            self.sc_index,
            self.sc_meshColors,
            R_bi,
            sc_position)
        window.addItem(self.sc_body)  # add spacecraft to plot     

    def update(self, state):
        sc_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        self.sc_body = self.update_object(
            self.sc_body,
            self.sc_points,
            self.sc_index,
            self.sc_meshColors,
            R_bi,
            sc_position)

    def add_object(self, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points, index)
        object = gl.GLMeshItem(
            vertexes=mesh,  # defines the triangular mesh (Nx3x3)
            vertexColors=colors,  # defines mesh colors (Nx1)
            drawEdges=True,  # draw edges between mesh elements
            smooth=False,  # speeds up rendering
            computeNormals=False)  # speeds up rendering
        return object

    def update_object(self, object, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points, index)
        object.setMeshData(vertexes=mesh, vertexColors=colors)
        return object

    def get_sc_points(self):
        """"
            Points that define the spacecraft, and the colors of the triangular mesh
            Define the points on the spacecraft following information in Appendix C.3
        """
        # points are in XYZ coordinates
        #   define the points on the spacecraft according to Appendix C.3

        unit_length = 1
        fuse_h = unit_length
        fuse_w = unit_length
        fuse_l1 = unit_length * 2
        fuse_l2 = unit_length
        fuse_l3 = unit_length * 4
        wing_l = unit_length
        wing_w = unit_length * 6
        tail_h = unit_length
        tail_l = unit_length
        tail_w = unit_length * 2


        points = self.unit_length * np.array([
            [fuse_l1, 0, 0],  # point 1 [0]
            [fuse_l2, fuse_w/2, -fuse_h/2],  # point 2 [1]
            [fuse_l2, -fuse_w/2, -fuse_h/2],  # point 3 [2]
            [fuse_l2, -fuse_w/2, fuse_h/2],  # point 4 [3]
            [fuse_l2, fuse_w/2, -fuse_h/2],  # point 5 [4]
            [fuse_l3, 0, 0],  # point 6 [5]
            [0, wing_w/2, 0],  # point 7 [6]
            [wing_l, wing_w/2, 0],  # point 8 [7]
            [wing_l, -wing_w/2, 0],  # point 9 [8]
            [0, -wing_w/2, 0],  # point 10 [9]
            [-(fuse_l3-tail_l), tail_w/2, 0],  # point 11 [10]
            [-fuse_l3, tail_w/2, 0],  # point 12 [11]
            [-fuse_l3, -tail_w/2, 0], # point 13 [12]
            [-(fuse_l3-tail_l), -tail_w/2, 0],  # point 14 [13]
            [-(fuse_l3-tail_l), 0, 0], # point 15 [14]
            [-fuse_l3, 0, -tail_h] # point 16 [15]
            ]).T
        # point index that defines the mesh
        index = np.array([
            [0, 1, 2],  # front top
            [0, 2, 3],  # front left
            [0, 3, 4],  # front bottom
            [0, 4, 1],  # front right
            [2, 3, 5], # body left
            [1, 2, 5], # body top
            [2, 4, 5], # body right  
            [4, 3, 5], # body bottom
            [6, 7, 8], # wing top
            [6, 8, 9], # wing bottom
            [10, 11, 12], # tail top
            [10, 12, 13], # tail bottom
            [5, 14, 15], # tail top
            ])
        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((13, 3, 4), dtype=np.float32)
        meshColors[0] = yellow  # front 1
        meshColors[1] = yellow  # front 2
        meshColors[2] = yellow  # back 1
        meshColors[3] = yellow  # back 2
        meshColors[4] = blue  # right 1
        meshColors[5] = blue  # right 2
        meshColors[6] = blue  # left 1
        meshColors[7] = blue  # left 2
        meshColors[8] = red  # top 1
        meshColors[9] = red  # top 2
        meshColors[10] = green  # bottom 1
        meshColors[11] = green  # bottom 2
        return points, index, meshColors



"""
example of drawing a box-like spacecraft in python
    - Beard & McLain, PUP, 2012
    - Update history:  
        1/8/2019 - RWB
"""
import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.Vector as Vector
from PyQt5 import QtWidgets,QtGui
import random
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.colors as mcolors


class mav_viewer():
    def __init__(self,x_obs,y_obs,z_obs,r_obs):
        # Initialize Qt application
        self.app = QtWidgets.QApplication([])  # Initialize QT

        # Create the main window
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle('Submarine Viewer')
        self.window.setGeometry(0, 0, 1400, 1000)  # Main window size

        # Create a central widget and layout
        self.central_widget = QtWidgets.QWidget()
        self.window.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QGridLayout()
        self.central_widget.setLayout(self.layout)


        # Flight Visualisation - Proj MPPI
        self.view_3d_proj = gl.GLViewWidget()
        self.view_3d_proj.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.view_3d_proj.setCameraPosition(distance=20)
        self.view_3d_proj.setBackgroundColor('k')
        self.layout.addWidget(self.view_3d_proj,0, 0)


        # Flight Visualisation - Base  MPPI
        self.view_3d_base = gl.GLViewWidget()
        self.view_3d_base.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.view_3d_base.setCameraPosition(distance=20)
        self.view_3d_base.setBackgroundColor('k')
        self.layout.addWidget(self.view_3d_base,1, 0)

        #  # Create a vertical layout for 2D plots
        self.plot_layout = QtWidgets.QVBoxLayout()
        # self.layout.addLayout(self.plot_layout,0, 0)

        # Create 2D plot widget for trajectory - Proj
        self.plot_2d_proj = pg.PlotWidget()
        self.plot_2d_proj.setTitle('Trajectory - Proj MPPI')
        self.plot_2d_proj.setLabel('left', 'y(m)')
        self.plot_2d_proj.setLabel('bottom', 'x(m)')
        for i in range(x_obs.shape[0]):
            self.add_circle_proj(y_obs[i], x_obs[i], r_obs[i])
        self.plot_layout.addWidget(self.plot_2d_proj)
        self.layout.addWidget(self.plot_2d_proj,0, 1)
        self.plot_data_proj = self.plot_2d_proj.plot([], [], pen=pg.mkPen('r', width=2))
        self.plot_2d_proj.setAspectLocked(True)  # Ensure equal scaling
        self.plot_2d_proj.setMaximumHeight(500)
        self.plot_2d_proj.setMaximumWidth(500) 
        

        # Create 2D plot widget for trajectory - Base
        self.plot_2d_base = pg.PlotWidget()
        self.plot_2d_base.setTitle('Trajectory - Base MPPI')
        self.plot_2d_base.setLabel('left', 'y(m)')
        self.plot_2d_base.setLabel('bottom', 'x(m)')
        for i in range(x_obs.shape[0]):
            self.add_circle_base(x_obs[i], y_obs[i], r_obs[i])
        
        self.plot_layout.addWidget(self.plot_2d_base)
        self.layout.addWidget(self.plot_2d_base,1, 1)
        self.plot_data_base = self.plot_2d_base.plot([], [], pen=pg.mkPen('r', width=2))
        self.plot_2d_base.setMaximumHeight(500)
        self.plot_2d_base.setMaximumWidth(500) 



        self.plot_initialized = False

        self.points, self.meshColors = self._get_spacecraft_points()
        self.points_proj = self.points
        self.points_base = self.points
        # Initialize the state for trajectory tracking
        self.trajectory_x_proj = []
        self.trajectory_y_proj = []

        self.trajectory_x_base = []
        self.trajectory_y_base = []

        # Add spheres to the 3D view
        self.add_spheres(x_obs, y_obs, z_obs, r_obs)

        # Display the window
        self.window.show()

    def add_spheres(self, x_obs, y_obs, z_obs, r_obs):
        for x, y, z, r in zip(x_obs, y_obs, z_obs, r_obs):
            sphere_data = gl.MeshData.sphere(rows=20, cols=20, radius=r)
            mesh = gl.GLMeshItem(meshdata=sphere_data, smooth=True, color=(1, 0, 0, 1), shader='shaded', drawEdges=True)
            mesh.translate(x, y, z)
            self.view_3d_proj.addItem(mesh)
            self.view_3d_base.addItem(mesh)

    def add_circle_proj(self, x, y, radius):
        ellipse = QtWidgets.QGraphicsEllipseItem(x - radius, y - radius, 2 * radius, 2 * radius)
        ellipse.setPen(QtGui.QPen(QtGui.QColor('blue'), 2))
        ellipse.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 50)))
        self.plot_2d_proj.addItem(ellipse)

    def add_circle_base(self, x, y, radius):
        ellipse = QtWidgets.QGraphicsEllipseItem(x - radius, y - radius, 2 * radius, 2 * radius)
        ellipse.setPen(QtGui.QPen(QtGui.QColor('blue'), 2))
        ellipse.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 50)))
        self.plot_2d_base.addItem(ellipse)

    def update(self, state_proj, state_base):
            
            ######################## Projected MPPI ########################

            spacecraft_position_proj = np.array([[state_proj.pn], [state_proj.pe], [-state_proj.h]])  # NED coordinates
            R_proj = self._Euler2Rotation(state_proj.phi, state_proj.theta, state_proj.psi)  # Attitude
            rotated_points_proj = self._rotate_points(self.points_proj, R_proj)
            translated_points_proj = self._translate_points(rotated_points_proj, spacecraft_position_proj)
            
            # Convert North-East Down to East-North-Up for rendering
            R_up_proj = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
            translated_points_proj = R_up_proj @ translated_points_proj
            
            # Convert points to triangular mesh
            mesh_proj = self._points_to_mesh(translated_points_proj)


            ######################## BASE MPPI #######################


            spacecraft_position_base = np.array([[state_base.pn], [state_base.pe], [-state_base.h]])  # NED coordinates
            R_base = self._Euler2Rotation(state_base.phi, state_base.theta, state_base.psi)  # Attitude
            rotated_points_base = self._rotate_points(self.points_base, R_base)
            translated_points_base = self._translate_points(rotated_points_base, spacecraft_position_base)
            
            # Convert North-East Down to East-North-Up for rendering
            R_up_base = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
            translated_points_base = R_up_base @ translated_points_base
            
            # Convert points to triangular mesh
            mesh_base = self._points_to_mesh(translated_points_base)


            # Initialisation of spacecraft


            # Initialize the drawing of the spacecraft
            if not self.plot_initialized:
                self.body_proj = gl.GLMeshItem(vertexes=mesh_proj,
                                        vertexColors=self.meshColors,
                                        drawEdges=True,
                                        smooth=False,
                                        computeNormals=False)
                self.view_3d_proj.addItem(self.body_proj)

                self.body_base = gl.GLMeshItem(vertexes=mesh_base,
                                        vertexColors=self.meshColors,
                                        drawEdges=True,
                                        smooth=False,
                                        computeNormals=False)
                self.view_3d_base.addItem(self.body_base)
                
                self.plot_initialized = True
            else:
                self.body_proj.setMeshData(vertexes=mesh_proj, vertexColors=self.meshColors)
                self.body_base.setMeshData(vertexes=mesh_base, vertexColors=self.meshColors)


            ############# Proj MPPI 3d
            view_location = Vector(state_proj.pe, state_proj.pn, state_proj.h)
            self.view_3d_proj.opts['center'] = view_location

            # Camera Position - Proj
            fixed_elevation = 10

            camera_distance = 60  # Fixed longitudinal distance 

            # Calculate the camera's position behind the object
            forward_vector_proj = R_proj[:, 2]
            camera_position_proj = Vector(state_proj.pe, state_proj.pn, state_proj.h) - camera_distance * forward_vector_proj

            target_azimuth_proj = np.arctan2(-R_proj[1, 1], R_proj[0, 1]) * 180 / np.pi

            # Update the camera view to follow the object
            self.view_3d_proj.opts['center'] = Vector(state_proj.pe, state_proj.pn, state_proj.h)
            self.view_3d_proj.opts['elevation'] = fixed_elevation
            self.view_3d_proj.opts['azimuth'] = target_azimuth_proj
            self.view_3d_proj.opts['distance'] = camera_distance

            ############# Base MPPI 3d
            view_location = Vector(state_base.pe, state_base.pn, state_base.h)
            self.view_3d_base.opts['center'] = view_location

            # Camera Position - Proj
            forward_vector_base = R_base[:, 2]
            camera_position_base = Vector(state_base.pe, state_base.pn, state_base.h) - camera_distance * forward_vector_base

            # Calculate the azimuth without affecting the elevation (pitch)
            target_azimuth_base = np.arctan2(-R_base[1, 1], R_base[0, 1]) * 180 / np.pi

            # Update the camera view to follow the object
            self.view_3d_base.opts['center'] = Vector(state_base.pe, state_base.pn, state_base.h)
            self.view_3d_base.opts['elevation'] = fixed_elevation
            self.view_3d_base.opts['azimuth'] = target_azimuth_base
            self.view_3d_base.opts['distance'] = camera_distance

            # Update trajectory data
            self.trajectory_x_proj.append(state_proj.pe)
            self.trajectory_y_proj.append(state_proj.pn)

            self.trajectory_x_base.append(state_base.pe)
            self.trajectory_y_base.append(state_base.pn)

            self.plot_data_proj.setData(self.trajectory_x_proj, self.trajectory_y_proj)

            self.plot_data_base.setData(self.trajectory_x_base, self.trajectory_y_base)
            
            # Redraw
            self.app.processEvents()

        ###################################
        # private functions
    def _rotate_points(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = R @ points
        return rotated_points

    def _translate_points(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation, np.ones([1,points.shape[1]]))
        return translated_points

    def _get_spacecraft_points(self):
        """"
            Points that define the spacecraft, and the colors of the triangular mesh
            Define the points on the aircraft following diagram in Figure C.3
        """
        # points are in NED coordinates
        bu= 1  # base unit for the plane
        # X coordinates
        fl1=2*bu
        fl2=1*bu
        fl3=4*bu
        wl=1*bu
        twl=1*bu
        # Y coordinates
        fw=1*bu
        ww=5*bu
        tww=3*bu
        # Z coordinates
        fh=1*bu
        th=1*bu

        points = np.array([[fl1, 0, 0],  # point 1
                            [fl2, fw/2, -fh/2],  # point 2
                            [fl2, -fw/2, -fh/2],  # point 3
                            [fl2, -fw/2, fh/2],  # point 4
                            [fl2, fw/2, fh/2],  # point 5
                            [-fl3, 0, 0],  # point 6
                            [0, ww/2, 0],  # point 7
                            [-wl, ww/2, 0],  # point 8
                            [-wl, -ww/2, 0],  # point 9
                            [0, -ww/2, 0],  # point 10
                            [-fl3+twl, tww/2, 0],  # point 11
                            [-fl3, tww/2, 0],  # point 12
                            [-fl3, -tww/2, 0],  # point 13
                            [-fl3+twl, -tww/2, 0],  # point 14
                            [-fl3+twl, 0, 0],  # point 15
                            [-fl3, 0, -th],  # point 16
                            ]).T
        # scale points for better rendering
        scale = 1
        points = scale * points

        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((13, 3, 4), dtype=np.float32)
        meshColors[0] = yellow  # noset
        meshColors[1] = yellow  # noser
        meshColors[2] = yellow  # noseb
        meshColors[3] = yellow  # nosel
        meshColors[4] = blue  # flt
        meshColors[5] = blue  # flr
        meshColors[6] = blue  # flb
        meshColors[7] = blue  # fll
        meshColors[8] = green  # wing
        meshColors[9] = green  # wing
        meshColors[10] = red  # tail
        meshColors[11] = red  # tail
        meshColors[12] = green  # rudder
        return points, meshColors

    def _points_to_mesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
            (a rectangle requires two triangular mesh faces)
        """
        points=points.T
        mesh = np.array([[points[0], points[1], points[2]],  # noset
                            [points[0], points[2], points[3]],  # noser
                            [points[0], points[3], points[4]],  # noseb
                            [points[0], points[4], points[1]],  # nosel
                            [points[5], points[1], points[2]],  # flt
                            [points[5], points[2], points[3]],  # flr
                            [points[5], points[3], points[4]],  # flb
                            [points[5], points[4], points[1]],  # fll
                            [points[6], points[7], points[8]],  # wing
                            [points[8], points[9], points[6]],  # wing
                            [points[10], points[11], points[12]],  # tail
                            [points[12], points[13], points[10]],  # tail
                            [points[5], points[14], points[15]],  # rudder
                            ])
        return mesh

    def _Euler2Rotation(self, phi, theta, psi):
        """
        Converts euler angles to rotation matrix (R_b^i, i.e., body to inertial)
        """
        # only call sin and cos once for each angle to speed up rendering
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        R_roll = np.array([[1, 0, 0],
                            [0, c_phi, s_phi],
                            [0, -s_phi, c_phi]])
        R_pitch = np.array([[c_theta, 0, -s_theta],
                            [0, 1, 0],
                            [s_theta, 0, c_theta]])
        R_yaw = np.array([[c_psi, s_psi, 0],
                            [-s_psi, c_psi, 0],
                            [0, 0, 1]])
        R = R_roll @ R_pitch @ R_yaw  # inertial to body (Equation 2.4 in book)
        return R.T  # transpose to return body to inertial











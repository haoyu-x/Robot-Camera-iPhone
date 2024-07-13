import numpy as np
from record3d import Record3DStream

from threading import Event

import numpy as np
import open3d as o3d

from PIL import Image
from quaternion import as_rotation_matrix, quaternion



import plotly.graph_objs as go
import plotly.io as pio
from scipy.spatial.transform import Rotation

import time

# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/realsense_pcd_visualizer.py

# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python








class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        self.rgb_width = 720
        self.rgb_height = 960

        self.init_camera_pose = None
        # Create a Visualizer object
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("iPhone Point Cloud Steaming", width=self.rgb_width, height=self.rgb_height)
        self.vis.get_view_control()

        # Create a PointCloud object
        self.pcd = o3d.geometry.PointCloud()



        # self.rerunapp = rerunio.Application()


    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])


    def reshape_depth_and_conf(self, depth_image, confidence, rgb_image):


        pil_depth = Image.fromarray(depth_image)
        reshaped_depth = pil_depth.resize((self.rgb_width, self.rgb_height))
        reshaped_depth = np.asarray(reshaped_depth)


        conf_img = Image.fromarray(confidence)
        reshaped_conf = conf_img.resize((self.rgb_width, self.rgb_height))
        reshaped_conf = np.asarray(reshaped_conf)
      
      
        rgb = Image.fromarray(rgb_image)
        reshaped_rgb = rgb.resize((self.rgb_width, self.rgb_height))
        reshaped_rgb = np.asarray(reshaped_rgb)
        

        return reshaped_depth, reshaped_conf, reshaped_rgb


    





    def create_wireframe_cube(self, size=1.0):
        points = [[-size, -size, -size],
                [-size, -size, size],
                [-size, size, -size],
                [-size, size, size],
                [size, -size, -size],
                [size, -size, size],
                [size, size, -size],
                [size, size, size]]

        lines = [[0, 1], [1, 3], [3, 2], [2, 0],
                [4, 5], [5, 7], [7, 6], [6, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        return line_set



    def pose_to_extrinsic_matrix(self, camera_pose):
        """
        Convert a pose to an extrinsic matrix.
        """

        extrinsic_matrix = np.eye(4)
        qx, qy, qz, qw, px, py, pz = camera_pose.qx, camera_pose.qy, camera_pose.qz, camera_pose.qw, camera_pose.tx, camera_pose.ty, camera_pose.tz
        extrinsic_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        extrinsic_matrix[:3, -1] = [px, py, pz]



        return extrinsic_matrix




       
    def start_processing_stream(self):

        frame_count = 0
        frames = []
        prev_pose_matrix = None
        
        while True:                  
                self.event.wait()  # Wait for new frame to arrive
                # Copy the newly arrived RGBD frame
                depth = self.session.get_depth_frame()
                rgb = self.session.get_rgb_frame()
                # print(rgb.shape)    (960, 720, 3)
                # print(depth.shape)   (256, 192)
                confidence = self.session.get_confidence_frame()

                depth, confidence, rgb = self.reshape_depth_and_conf(depth, confidence, rgb)
                depth = depth.copy()
                rgb = rgb.copy()


                # this is the camera pose reading from iphone with respect to the world frame,
                # but the world frame is not the same as the inial frame camera frame when this script starts
                camera_pose = self.session.get_camera_pose() 
                extrinsic = self.pose_to_extrinsic_matrix(camera_pose)



            


                if self.init_camera_pose is None:




                    self.init_camera_pose = extrinsic

                    # define world frame as the initial camera pose
                    self.init_camera_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    self.vis.add_geometry(self.init_camera_frame_mesh.transform(self.init_camera_pose))
                    # add a cube as the work space
                    cube = self.create_wireframe_cube(size=1)
                    self.vis.add_geometry(cube.transform(self.init_camera_pose))


                    self.camera_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    self.vis.add_geometry(self.camera_frame_mesh.transform(self.init_camera_pose))


                else:
                                
             
             
                    self.vis.update_geometry(self.camera_frame_mesh.transform(np.linalg.inv(prev_pose_matrix)))

                    self.vis.update_geometry(self.camera_frame_mesh.transform(extrinsic))


                    
                    # Update visualization
                    self.vis.poll_events()
                    self.vis.update_renderer()



                frame_count += 1
                prev_pose_matrix = extrinsic

                self.event.clear()

   

if __name__ == '__main__':
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()

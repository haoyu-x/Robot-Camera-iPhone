import numpy as np
from record3d import Record3DStream
import cv2
from threading import Event

import numpy as np
import open3d as o3d

from PIL import Image
from quaternion import as_rotation_matrix, quaternion

import time


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


import pyrealsense2 as rs
import numpy as np
from enum import IntEnum

from datetime import datetime
import open3d as o3d

from os.path import abspath
import sys
sys.path.append(abspath(__file__))






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

    def rgbd_visualization(rgb, depth):

        # rgbd visualizer with opencv
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=200), cv2.COLORMAP_JET)
        # Show the RGBD Stream
        cv2.imshow('RGB', rgb)
        cv2.imshow('Depth', depth_colormap)

        # if confidence.shape[0] > 0 and confidence.shape[1] > 0:
        #     cv2.imshow('Confidence', confidence * 100)
        cv2.waitKey(1)

    

    def plotly_point_cloud_video(self, pcds):
        
        downsampled_pcds = []
        for pcd in pcds:
            every_k_points = 10  # Adjust the k value as needed
            downsampled_pcd = pcd.uniform_down_sample(every_k_points)
            downsampled_pcds.append(downsampled_pcd)

        pcds = downsampled_pcds
        num_frames = len(pcds)
        frames = []
        
        # Define bounding box coordinates (example values, adjust as needed)
        bbox = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ])

        # Define the edges of the bounding box
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]

        for i, pcd in enumerate(pcds):
            xyz = np.asarray(pcd.points)

            colors = np.asarray(pcd.colors) * 255
            color_strings = ['rgb({},{},{})'.format(r, g, b) for r, g, b in colors]

            # Create scatter plot for point cloud
            point_cloud_trace = go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode='markers',
                marker=dict(size=5, color=color_strings, opacity=0.8)
            )

            # Create scatter plots for bounding box edges
            bbox_traces = []
            for edge in edges:
                x_coords = [bbox[edge[0], 0], bbox[edge[1], 0]]
                y_coords = [bbox[edge[0], 1], bbox[edge[1], 1]]
                z_coords = [bbox[edge[0], 2], bbox[edge[1], 2]]
                bbox_trace = go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='lines',
                    line=dict(color='red', width=2)
                )
                bbox_traces.append(bbox_trace)

            frames.append(go.Frame(data=[point_cloud_trace] + bbox_traces, name=str(i)))

        # Initialize the figure with the first frame's data
        initial_xyz = np.asarray(pcds[0].points)
        initial_colors = np.asarray(pcds[0].colors) * 255
        initial_color_strings = ['rgb({},{},{})'.format(r, g, b) for r, g, b in initial_colors]

        initial_point_cloud_trace = go.Scatter3d(
            x=initial_xyz[:, 0],
            y=initial_xyz[:, 1],
            z=initial_xyz[:, 2],
            mode='markers',
            marker=dict(size=5, color=initial_color_strings, opacity=0.8)
        )

        initial_bbox_traces = []
        for edge in edges:
            x_coords = [bbox[edge[0], 0], bbox[edge[1], 0]]
            y_coords = [bbox[edge[0], 1], bbox[edge[1], 1]]
            z_coords = [bbox[edge[0], 2], bbox[edge[1], 2]]
            bbox_trace = go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                line=dict(color='red', width=2)
            )
            initial_bbox_traces.append(bbox_trace)

        figure = go.Figure(
            data=[initial_point_cloud_trace] + initial_bbox_traces,
            layout=go.Layout(
                title='Animating Point Cloud with Plotly',
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {
                                'duration': 500,
                                'redraw': True
                            },
                            'fromcurrent': True
                        }]
                    }]
                }],
                sliders=[{
                    'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {
                        'font': {'size': 20},
                        'prefix': 'Frame:',
                        'visible': True,
                        'xanchor': 'right'
                    },
                    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                    'pad': {'b': 10},
                    'len': 0.9,
                    'x': 0.1,
                    'y': 0,
                    'steps': [{'label': str(frame), 'method': 'animate', 'args': [[str(frame)], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate'}]} for frame in range(num_frames)]
                }]
            ),
            frames=frames
        )

        # camera = dict(
        #     eye=dict(x=1.5, y=1.5, z=1.5),  # Camera position
        #     center=dict(x=0, y=0, z=0),     # Center of rotation
        #     up=dict(x=0, y=0, z=1)          # Up direction
        # # )
        # figure.update_layout(scene_camera=camera)

        pio.write_html(figure, "iphone_point_cloud_animation.html", auto_open=True)



    def plotly_point_cloud(self, pointclouds):

        pointclouds.transform([
                                [1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]
                            ])


        every_k_points = 10  # Adjust the k value as needed
        pointclouds = pointclouds.uniform_down_sample(every_k_points)


        # Assuming you already have the color point cloud `temp`
        # Extract points and colors
        points = np.asarray(pointclouds.points)
        colors = np.asarray(pointclouds.colors)*255

        x = points[:, 0]
        print(x.shape)  

        # Use the z-coordinate as color
        depth = points[:, 2]

        # Normalize depth values to be in the range [0, 1] for coloring
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())


        color_strings = ['rgb({},{},{})'.format(r, g, b) for r, g, b in colors]


        # Create a scatter plot with Plotly
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=color_strings,       
                colorscale='Viridis',   # Choose a colorscale
                opacity=0.8
            )
        )])





        # Update layout for better visibility
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(title='X Axis'),
                yaxis=dict(title='Y Axis'),
                zaxis=dict(title='Z Axis')
            )
        )


        # self.visualizer.update_bounds(self.workspace_bounds_min, self.workspace_bounds_max)

        # def update_bounds(self, lower, upper):
        #     self.workspace_bounds_min = lower
        #     self.workspace_bounds_max = upper
        #     self.plot_bounds_min = lower - 0.15 * (upper - lower)
        #     self.plot_bounds_max = upper + 0.15 * (upper - lower)
        #     xyz_ratio = 1 / (self.workspace_bounds_max - self.workspace_bounds_min)
        #     scene_scale = np.max(xyz_ratio) / xyz_ratio
        #     self.scene_scale = scene_scale

        # # set bounds and ratio
        # fig.update_layout(scene=dict(xaxis=dict(range=[self.plot_bounds_min[0], self.plot_bounds_max[0]], autorange=False),
        #                             yaxis=dict(range=[self.plot_bounds_min[1], self.plot_bounds_max[1]], autorange=False),
        #                             zaxis=dict(range=[self.plot_bounds_min[2], self.plot_bounds_max[2]], autorange=False)),
        #                 scene_aspectmode='manual',
        #                 scene_aspectratio=dict(x=self.scene_scale[0], y=self.scene_scale[1], z=self.scene_scale[2]))

        # do not show grid and axes
        # fig.update_layout(scene=dict(
        #     xaxis=dict(showgrid=False,showticklabels=False, title='', visible=False, ) ,                      
        #     yaxis=dict(showgrid=False, showticklabels=False, title='', visible=False,),
        #     zaxis=dict(showgrid=False, showticklabels=False, title='', visible=False, ),
            
        #     ))


        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(title='X Axis', range=[-1, 1]),
                yaxis=dict(title='Y Axis', range=[-1, 1]),
                zaxis=dict(title='Z Axis', range=[-1, 1])
            )
        )
        # Show the plot
        fig.show()

        # Save the plot as an interactive HTML file
        # pio.write_html(fig, file='point_cloud_scatter_plot.html', auto_open=True)

        # # Visualize the point cloud using open3d
        # o3d.visualization.draw_geometries([point_cloud],
        #                                 window_name="Open3D Point Cloud",
        #                                 width=800,
        #                                 height=600,
        #                                 left=50,
        #                                 top=50,
        #                                 point_show_normal=False)



    def open3d_visualization(self, pcd, frame_count, extrinsic_matrix):

        if frame_count == 0:
            self.vis.add_geometry(pcd)
        

        self.vis.update_geometry(pcd)
        self.vis.poll_events()
        self.vis.update_renderer()


    def open3d_visualization_camera(self, pcd, frame_count, extrinsic_matrix, init_matrix=None):


        if frame_count == 0:
            self.vis.add_geometry(pcd)
            self.vis.add_geometry(self.mesh_frame)



        # Update the pose of the camera frame
        #self.mesh_frame.transform(extrinsic_matrix)
        #self.mesh_frame.transform(np.linalg.inv(init_matrix))

        # Update point cloud geometry
        self.vis.update_geometry(pcd)
        self.vis.update_geometry(self.mesh_frame)



        # Update renderer and poll events
        self.vis.poll_events()
        self.vis.update_renderer()

    def open3d_visualization_camera_debug_pose(self, extrinsic_matrix, prev_pose_matrix):


        print(prev_pose_matrix)
        print(np.linalg.inv(prev_pose_matrix))
        self.vis.update_geometry(self.camera_frame_mesh.transform(np.linalg.inv(prev_pose_matrix)))
        self.vis.update_geometry(self.camera_frame_mesh.transform(extrinsic_matrix))


        
        # Update visualization
        self.vis.poll_events()
        self.vis.update_renderer()





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

    def get_global_xyz(self, depth, rgb, confidence, intrinsics, camera_pose, depth_scale=1000.0, only_confident=False):

        # If only confident, replace not confident points with nans
        if only_confident:
            depth[confidence != 2] = np.nan


        depth_o3d = o3d.geometry.Image(
            np.ascontiguousarray(depth_scale * depth).astype(np.float32)
        )
        rgb_o3d = o3d.geometry.Image(
            np.ascontiguousarray(rgb).astype(np.uint8)
        )

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(self.rgb_width),
            height=int(self.rgb_height),
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
        )


        temp = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intrinsics)
        temp.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.pcd.points = temp.points
        self.pcd.colors = temp.colors


        # Now transform everything by camera pose to world frame.
        extrinsic_matrix =self.pose_to_extrinsic_matrix(camera_pose)
        self.pcd.transform(extrinsic_matrix)

        self.pcd.transform(np.linalg.inv(self.init_camera_pose))





       
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

                    # add a cube as the work space
                    self.vis.add_geometry(self.create_wireframe_cube(size=1))

                    intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
                    self.init_camera_pose = extrinsic

                    # define world frame as the initial camera pose
                    self.world_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    self.vis.add_geometry(self.world_frame_mesh)
                    self.init_camera_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    self.vis.add_geometry(self.world_frame_mesh.transform(self.init_camera_pose))

                    self.camera_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    self.vis.add_geometry(self.camera_frame_mesh.transform(self.init_camera_pose))





                else:
                    self.get_global_xyz(depth, rgb, confidence, intrinsic_mat, camera_pose, depth_scale=1000.0, only_confident=False)
                    frames.append(self.pcd)

                    time.sleep(1/60)


                    
                    self.open3d_visualization_camera_debug_pose(extrinsic, prev_pose_matrix)
                    # self.open3d_visualization(self.pcd, frame_count, extrinsic)
                    # self.open3d_visualization_camera(self.pcd, frame_count, extrinsic, init_matrix=self.world_extrinsic)

                    # self.rgbd_visualization(rgb, depth)
                frame_count += 1
                prev_pose_matrix = extrinsic

                self.event.clear()

        # self.plotly_point_cloud_video(frames)
        #for frame in frames:
        #    self.plotly_point_cloud(frame)  # Visualize the point cloud using Plotly


if __name__ == '__main__':
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()

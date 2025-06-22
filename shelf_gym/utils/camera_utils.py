import numpy as np
import pybullet as pb
import open3d as o3d
import time
import pdb
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.near, self.far = 0.07, 1.5
        self.fov = 58
        self.aspect = self.width / self.height

        self.cx = float(self.width) / 2.
        self.cy = float(self.height) / 2.

        self.intrinsic_matrix = None
        self.view_matrix = None
        self.projection_matrix = None

        self.camera_array_base = np.array([0, 0.95, 0.97 + 0.1])
        #mgrid is a meshgrid that is used to calculate the pixel coordinates of the depth image for the point cloud
        self.mgrid = np.mgrid[-1:1:2 / self.height, -1:1:2 / self.width]

        #empty list to store the camera params later
        self.camera_lines = []
        self.transformed_coord_list = []
        self.instance_to_class_dict = {}
        self.instance_to_class_conversion = []

    def get_view_matrix(self, x, y, z, target_x, target_y, target_z, init_ori=[0, 0, 0], client_id=0):
        # Compute direction vector from camera position to target
        target = np.array([target_x, target_y, target_z])
        camera_position = np.array([x, y, z])
        direction = target - camera_position
        direction = direction / np.linalg.norm(direction)  # Normalize the direction vector

        # Compute up vector (assuming global up is along the Y-axis)
        up_vector = np.array([0, 1, 0])  # Assuming Y is up, adjust if needed

        # Use the cross product to find the right vector (right-handed system)
        right_vector = np.cross(up_vector, direction)
        right_vector = right_vector / np.linalg.norm(right_vector)  # Normalize right vector

        # Recompute up vector to ensure it's perpendicular to both direction and right vectors
        up_vector = np.cross(direction, right_vector)
        up_vector = up_vector / np.linalg.norm(up_vector)  # Normalize up vector

        # Create a rotation matrix that aligns the camera's forward direction with the direction vector
        forward = -direction  # Camera's forward vector points opposite of the direction vector

        # Rotation matrix from the forward, right, and up vectors
        rotation_matrix = np.column_stack((right_vector, up_vector, forward))

        # Use this rotation matrix to compute the camera's orientation (yaw, pitch, roll)
        # We assume that the camera's initial orientation is facing along the -Z axis (standard for many applications)
        euler_angles = R.from_matrix(rotation_matrix).as_euler("XYZ", degrees=False)

        # Convert euler angles back to yaw, pitch, roll
        yaw, pitch, roll = np.rad2deg(euler_angles)

        # Compute view matrix using the new orientation
        return pb.computeViewMatrix(cameraEyePosition=camera_position,
                                    cameraTargetPosition=target,
                                    cameraUpVector=up_vector,
                                    physicsClientId=client_id), target


    # def get_view_matrix(self, x, y, z, yaw, pitch, roll, init_ori=[0,0,0], client_id=0):
    #     # Convert degrees to radians
    #     ori = init_ori + np.array([np.deg2rad(pitch), np.deg2rad(roll), np.deg2rad(yaw)])
    #     r = np.array(R.from_euler("XYZ", ori, degrees=False).as_matrix())
    #     init_camera_vector = np.array([0, 0, -1])  # Forward direction (Z-axis)
    #     init_up_vector = np.array([0, -1, 0])  # Up direction (Y-axis)
    #     # Rotate the vectors
    #     camera_vector = r.dot(init_camera_vector)
    #     up_vector = r.dot(init_up_vector)
    #     # Compute target position
    #     target = np.array([x, y, z]) + camera_vector * 0.1  # Slightly forward
    #     camera_position = np.array([x, y, z])
    #     return pb.computeViewMatrix(cameraEyePosition=camera_position,
    #                                 cameraTargetPosition=target,
    #                                 cameraUpVector=up_vector,
    #                                 physicsClientId=client_id), target


    def get_cam_in_hand(self, robot_id, camera_link, remove_gripper=True, client_id=0, no_conversion = False):
        '''
        Get the image from the camera in the hand of the robot
        Args: robot_id: id of the robot
                camera_link: link of the camera
                remove_gripper: whether to remove the gripper from the image through naive bounding box removal
                client_id: id of the pybullet client
                no_conversion: whether to convert the instance ids to class ids
        Returns: [rgb, depth, semantics, pointcloud, transformed_depth]
        '''

        # Center of mass position and orientation (of link-7)
        self.projection_matrix = pb.computeProjectionMatrixFOV(self.fov,
                                                               self.aspect,
                                                               self.near,
                                                               self.far,
                                                               physicsClientId=client_id)
        com_p, com_o, _, _, _, _ = pb.getLinkState(robot_id,
                                                   camera_link,
                                                   computeForwardKinematics=True,
                                                   physicsClientId=client_id)
        rot_matrix = pb.getMatrixFromQuaternion(com_o, physicsClientId=client_id)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (0, 1, 0)  # y-axis
        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        self.view_matrix = pb.computeViewMatrix(com_p, com_p + 0.1 * camera_vector,
                                                up_vector, physicsClientId=client_id)
        return self.get_image(remove_gripper=remove_gripper,no_conversion = no_conversion)


    def get_image(self, images=None, remove_gripper=False, view_matrix=None,
                  projection_matrix=None, intrinsic_matrix=None, client_id=0,
                  no_conversion=False):
        '''
        Get the image from the camera
        Args:
            images:
            remove_gripper: remove the gripper from the image
            view_matrix: precomputed view matrix
            projection_matrix: precomputed projection matrix
            intrinsic_matrix: precomputed intrinsic matrix
            client_id: pybullet client id
            no_conversion: whether to convert the instance ids to class ids
        Returns:
            Dictionary containing the rgb, depth, semantics, pointcloud, and transformed depth
        '''
        # Use provided matrices or default ones
        view_matrix = view_matrix if view_matrix is not None else self.view_matrix
        projection_matrix = projection_matrix if projection_matrix is not None else self.projection_matrix
        # Fetch camera image if not provided
        if images is None:
            images = pb.getCameraImage(self.width, self.height, view_matrix, projection_matrix,
                                       shadow=False, renderer=pb.ER_BULLET_HARDWARE_OPENGL,
                                       physicsClientId=client_id)

        # Extract RGB, Depth, and Semantics
        rgb = np.array(images[2])[:, :, :3]
        depth = images[3]
        self.semantics = images[4]

        # Assign unclassified items (ID=-1) to background ID (5000)
        self.semantics = np.where(self.semantics == -1, 5000, self.semantics)

        # Perform instance ID to class conversion if enabled
        if not no_conversion:
            max_id = np.max(self.semantics)
            if max_id >= len(self.instance_to_class_conversion):
                raise ValueError(
                    "Instance-to-class dictionary is missing an ID present in this image. "
                    "Ensure it is updated with the correct mappings."
                )

            class_mask = self.instance_to_class_conversion[self.semantics.astype(np.int32)]
            self.semantics = np.dstack([self.semantics, class_mask])

        # Remove gripper if needed
        if remove_gripper:
            depth = self.remove_gripper(depth)

        # Generate point cloud
        o3d_pcd, np_pcd = self.get_pointcloud(depth, projection_matrix, view_matrix, intrinsic_matrix)

        # Flip point cloud direction
        np_pcd *= -1

        # Add fringe binary mask
        new_pcd = np.pad(np_pcd, ((0, 0), (0, 0), (0, 1)), mode='constant')
        new_pcd[0, :, 3] = 1
        new_pcd[-1, :, 3] = 1
        new_pcd[:, 0, 3] = 1
        new_pcd[:, -1, 3] = 1

        # Depth transformation
        transformed_depth = (2.0 * self.near * self.far) / (
                    self.far + self.near - (2.0 * depth - 1.0) * (self.far - self.near))
        transformed_depth = np.round(transformed_depth * 1000).astype(np.uint16)

        # Return as dictionary
        return {
            "rgb": rgb,
            "depth": depth,
            "transformed_depth": transformed_depth,
            "semantics": self.semantics,
            "point_cloud": {
                "open3d": o3d_pcd,
                "numpy": new_pcd}
        }


    def remove_gripper(self, depth, open=False):
        '''
        Remove the gripper from the depth image
        Args:
            depth: depth image
            open: whether the gripper is open
        Returns: depth image with gripper removed
        '''

        if open:
            depth[193:, :53] = 1. #0.99
            depth[193:, 199:] = 1. #0.99
        else:
            if(len(self.semantics.shape)>2):
                instances = self.semantics[:,:,0]
            else:
                instances = self.semantics
            depth[instances == 0] = 1. #0.99
            depth[instances == 0] = 1. #0.99
        return depth


    def get_pointcloud(self, depth, projection_matrix, view_matrix, intrinsic_matrix=None, open3d=True):
        '''
        Get the point cloud from depth image
        Args:
            depth: current depth image
            projection_matrix: precomputed projection matrix
            view_matrix: precomputed view matrix
            intrinsic_matrix: precomputed intrinsic matrix
            open3d: whether to return point cloud in open3d format
        Returns: point cloud in open3d and numpy format
        '''

        if not open3d:
            pc = self.depth_to_point(depth, projection_matrix, view_matrix)
            pc = np.array(pc).reshape(self.width,  self.height, 3)
            return None, pc

        #transform opengl depth to linear open3d depth
        depth = self.depth_to_o3d(depth)
        if intrinsic_matrix is None:
            intrinsic_matrix = self.projection_matrix_to_intrinsic(projection_matrix)

        # transform view matrix to open3d convention
        vm = self.ogl_vm_to_o3d(view_matrix)

        # calculate open3d point cloud from intrinsics and depth image
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width,
                                                      self.height,
                                                      intrinsic_matrix[0, 0],
                                                      intrinsic_matrix[1, 1],
                                                      self.cx,
                                                      self.cy)
        pcd = o3d.geometry.PointCloud()
        o3d_pcd = pcd.create_from_depth_image(o3d.geometry.Image(depth), intrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1,project_valid_depth_only = False)

        # transform point cloud to world coordinates
        o3d_pcd.transform(vm)
        # numpy version of point cloud
        np_pcd = np.asarray(o3d_pcd.points).reshape(self.height,  self.width, 3, order="C")
        return o3d_pcd, np_pcd

    def depth_to_o3d(self, depth):
        '''
        convert normalised depth to real world depth in meters
        '''

        return (2.0 * self.near * self.far) / (self.far + self.near - (2.0 * depth - 1.0) * (self.far - self.near))

    def depth_to_point(self, depth, projection_matrix=None, view_matrix=None):
        # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        if projection_matrix is None:
            projection_matrix = self.projection_matrix
        if view_matrix is None:
            view_matrix = self.view_matrix
        proj_matrix = np.asarray(projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))
        # create a grid with pixel coordinates and depth values
        y, x = self.mgrid.copy()
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)
        pixels = np.stack([x, y, z, h], axis=1)

        pixels[:, 2] = 2 * pixels[:, 2] - 1
        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]
        return points


    def get_cam_pos_in_world(self):
        '''
        Get the camera position in world coordinates
        '''
        cam_pose = np.linalg.inv(np.array(self.view_matrix).reshape(4, 4).T)
        cam_pose[:, 1:3] = -cam_pose[:, 1:3]
        return cam_pose


    def get_target_pos_from_cam_to_world(self, cam_point):
        '''
        get a target position in world coordinates from camera pixel coordinates
        Args:
            cam_point: pixel coordinates of the target
        Returns:
            target position in world coordinates
        '''

        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        trans_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))
        # turn pixels to world coordinates
        cam_point_homogeneous = np.append(cam_point, 1.)
        points = np.matmul(trans_pix_world, cam_point_homogeneous.T).T
        points /= points[-1]
        return points[:3]


    def get_target_pos_from_world_to_cam(self, point):
        '''
        Get the target position in camera pixel coordinates from world coordinates
        Args:
            point: world coordinates of the target
        Returns: pixel coordinates of the target
        '''

        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        trans_cam = np.matmul(proj_matrix, view_matrix)
        ps_homogeneous = np.append(point, 1.)
        ps_transformed = np.matmul(trans_cam, ps_homogeneous.T).T
        ps_transformed /= ps_transformed[-1]
        return ps_transformed[:3]


    def check_if_pix_in_bounds(self, pix, w, h):
        '''
        quickly check if a pixel is in bounds of image and set it to the closest pixel if it is not
        '''

        if pix[0] < 0:
            pix[0] = 0
        if pix[0] > w-1:
            pix[0] = w-1
        if pix[1] < 0:
            pix[1] = 0
        if pix[1] > h-1:
            pix[1] = h-1
        return np.array(pix)


    def update_instance_to_class_dict(self,instance_to_class_dict):
        '''
        Update the instance to class dictionary
        Args:
            instance_to_class_dict: current instance to class dictionary
        Returns: updated instance to class dictionary
        '''

        self.instance_to_class_dict = instance_to_class_dict
        max_instance = np.max(list(self.instance_to_class_dict.keys()))
        self.instance_to_class_conversion = np.zeros(max_instance+1)
        for i in self.instance_to_class_dict.keys():
            self.instance_to_class_conversion[i] = self.instance_to_class_dict[i]


    '''
    #################################
        opengl to open3d  functions
    #################################
    '''


    def projection_matrix_to_intrinsic(self, projection_matrix):
        '''
        Convert Opengl projection matrix to intrinsic matrix in o3d format
        Args:
            projection_matrix: opengl projection matrix
        Returns: general intrinsic matrix
        '''

        # Calculate focal lengths and principal points from projection matrix to convert from opengl to open3d
        temp_matrix = np.asarray(projection_matrix).reshape([4, 4], order="F")
        n = temp_matrix[3, 2] / (temp_matrix[2, 2] + 1)
        r = n / temp_matrix[0, 0]
        t = n / temp_matrix[1, 1]
        fx = (self.width / 2) * (n / r)
        fy = (self.height / 2) * (n / t)

        intrinsic_matrix = np.array([[fx, 0, float(self.width) / 2],
                                     [0, fy, float(self.height) / 2],
                                     [0, 0, 1]])
        return intrinsic_matrix


    def ogl_vm_to_o3d(self, view_matrix, invert=True):
        '''
        Convert Opengl view matrix to open3d view matrix
        Args:
            view_matrix: opengl view matrix
            invert: whether to invert the transformation
        Returns: open3d view matrix
        '''

        vm = np.linalg.inv(np.transpose(np.reshape(np.array(view_matrix), (4, 4))))
        rotate_yz180 = np.array([[1, 0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, -1, 0],
                                 [0, 0, 0, 1]])
        if not invert:
            return vm @ rotate_yz180
        return rotate_yz180 @ vm @ np.linalg.inv(rotate_yz180)


    def opengl_to_o3d(self, projection_matrix, view_matrix):
        '''
        Convert Opengl projection and view matrices to open3d format
        Args:
            projection_matrix: opengl projection matrix
            view_matrix: opengl view matrix
        Returns: transformed view matrix and intrinsic matrix in open3d format
        '''

        # Calculate focal lengths and principal points from projection matrix to convert from opengl to open3d
        intrinsic_matrix = self.projection_matrix_to_intrinsic(projection_matrix)
        transformed_view_matrix = self.ogl_vm_to_o3d(view_matrix)

        return np.linalg.inv(transformed_view_matrix), intrinsic_matrix


    '''
    #################################
        Debug functions
    #################################
    '''


    def draw_cameras(self, cameras, target_point=[0.3, 0.85, 1.07], client_id=0):
        '''
        Draw debug lines from cameras to target point in pybullet
        '''

        line_length = 0.1  # Length of the debug line
        line_color = [1, 0, 0]  # Red color for the debug line

        for camera in cameras:
            start_point = camera["position"]
            end_point = start_point + line_length * (target_point - start_point) / np.linalg.norm(target_point - start_point)
            self.camera_lines.append(pb.addUserDebugLine(start_point, end_point, line_color, physicsClientId=client_id))


    def draw_gt_cameras(self, position, client_id=0):
        '''
        Draw debug lines from cameras to target point in pybullet for groundtruth cameras
        '''
        line_length = 0.1  # Length of the debug line
        line_color = [1, 0, 0]  # Red color for the debug line
        start_point = position
        position[-1] = 1.3
        end_point = start_point + line_length *np.array([0,0,-1])
        self.camera_lines.append(pb.addUserDebugLine(start_point, end_point, line_color, physicsClientId=client_id))


    def remove_cameras(self, client_id=0):
        '''
        Remove all debug lines from pybullet
        '''
        pb.removeAllUserDebugItems(physicsClientId=client_id)
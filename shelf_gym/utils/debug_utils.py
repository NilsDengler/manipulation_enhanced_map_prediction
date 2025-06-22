import pybullet as pb
import numpy as np

class SphereMarker:
    '''
    Generate a debug sphere marker in the pybullet environment
    '''
    def __init__(self, position, radius=0.05, rgba_color=(1, 0, 0, 0.8), text=None, orientation=None, client_id=0):
        self.client_id = client_id
        position = np.array(position)
        vs_id = pb.createVisualShape(
            pb.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color, physicsClientId=self.client_id)

        self.marker_id = pb.createMultiBody(
            baseMass=0,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vs_id,
            basePosition=position,
            useMaximalCoordinates=False,
            physicsClientId=self.client_id
        )

        self.debug_item_ids = list()
        if text is not None:
            self.debug_item_ids.append(
                pb.addUserDebugText(text, position + radius, physicsClientId=self.client_id)
            )

        if orientation is not None:
            # x axis
            axis_size = 2 * radius
            rotation_mat = np.asarray(pb.getMatrixFromQuaternion(orientation, physicsClientId=self.client_id)).reshape(3, 3)

            # x axis
            x_end = np.array([[axis_size, 0, 0]]).transpose()
            x_end = np.matmul(rotation_mat, x_end)
            x_end = position + x_end[:, 0]
            self.debug_item_ids.append(
                pb.addUserDebugLine(position, x_end, lineColorRGB=(1, 0, 0), physicsClientId=self.client_id)
            )
            # y axis
            y_end = np.array([[0, axis_size, 0]]).transpose()
            y_end = np.matmul(rotation_mat, y_end)
            y_end = position + y_end[:, 0]
            self.debug_item_ids.append(
                pb.addUserDebugLine(position, y_end, lineColorRGB=(0, 1, 0), physicsClientId=self.client_id)
            )
            # z axis
            z_end = np.array([[0, 0, axis_size]]).transpose()
            z_end = np.matmul(rotation_mat, z_end)
            z_end = position + z_end[:, 0]
            self.debug_item_ids.append(
                pb.addUserDebugLine(position, z_end, lineColorRGB=(0, 0, 1), physicsClientId=self.client_id)
            )

    def __del__(self):
        pb.removeBody(self.marker_id, physicsClientId=self.client_id)
        for debug_item_id in self.debug_item_ids:
            pb.removeUserDebugItem(debug_item_id, physicsClientId=self.client_id)
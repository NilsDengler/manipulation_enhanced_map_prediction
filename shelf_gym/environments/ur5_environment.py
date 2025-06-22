#!/usr/bin/python3
import numpy as np
from collections import deque
import os
from shelf_gym.environments.base_environment import BasePybulletEnv
from shelf_gym.utils.robot_utils import setup_robot
from shelf_gym.utils.klampt_utils import KlamptUtils
import pdb

class RobotEnv(BasePybulletEnv):
    DEFAULT_PATH = os.path.join(os.path.dirname(__file__), '../')
    def __init__(self,render=False, shared_memory=False, hz=240, show_vis=False):
        super().__init__(render=render, shared_memory=shared_memory, hz=hz)

        #define robot limits and initial parameters
        self.gripper_open_limit = (0.0, 0.085)
        self.default_gripper_height = 0.97

        self.ee_position_limit = ((-0.4, 0.4), (0.3, 0.9), (0.8, 1.4))
        self.initial_parameters = (1.1297861623268122, -1.4717860300352437, 2.5400780616664034,
                                   -1.068276383927301, 1.1306038004860783, 0.0015930864785381922, 0.)

        # create robot entity in environment and initialize
        self.load_robot()
        self.joints, self.controlGripper, self.controlJoints, self.mimicParentName = setup_robot(self._p, self.robot_id)
        self.reset_robot(self.initial_parameters)
        self.initialize_gripper()

        # save important link IDs to variables
        self.eef_id = self.joints['ee_fixed_joint'][0]
        self.tool_tip_id = self.joints['tool0_fixed_joint-tool_tip'][0]
        self.grasp_link_id = self.joints['grasp_joint'][0]
        self.tool_tip_left_id = self.joints['left_outer_finger_joint'][0]
        self.tool_tip_right_id = self.joints['right_outer_finger_joint'][0]
        self.camera_link = self.joints['dummy_camera_joint'][0]

        # Store initial poses of tool_tip and camera
        link_state = self._p.getLinkState(self.robot_id, self.tool_tip_id)
        self.init_pos = list(link_state[0])
        self.init_ori = np.array(self._p.getEulerFromQuaternion(link_state[1]))

        camera_link_state = self._p.getLinkState(self.robot_id, self.camera_link)
        self.camera_init_pos = list(camera_link_state[0])
        self.camera_init_ori = np.array(self._p.getEulerFromQuaternion(camera_link_state[1]))

        # If you can derive this from `self.joints` or another method, it would be preferable
        self.arm_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.gripper_joint_names = ["finger_joint", "left_outer_finger_joint", "left_inner_finger_joint",
                                    "left_inner_finger_pad_joint", "left_inner_knuckle_joint",
                                    "right_outer_knuckle_joint",  "right_outer_finger_joint",
                                    "right_inner_finger_joint",  "right_inner_finger_pad_joint",
                                    "right_inner_knuckle_joint"]

        #self.arm_joint_indices = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21]
        self.arm_joint_indices = [self.joints[n][0] for n in self.arm_joint_names]
        self.gripper_joint_indices =[self.joints[n][0] for n in self.gripper_joint_names]
        self.arm_and_gripper_joint_indices = self.arm_joint_indices + self.gripper_joint_indices
        self.init_arm_and_gripper_joint_config = self.get_current_arm_and_gripper_joint_config()

        # Initialize Klampt as motion planner
        self.show_vis = show_vis
        self.klampt_utils = KlamptUtils(self.DEFAULT_PATH, self.init_arm_and_gripper_joint_config, show_vis=self.show_vis)
        self.control_settings_dict = {'position_gain':{'free':0.03,'pushing':0.005}}


    def initialize_gripper(self):
        """
        Initialize the gripper with force sensors and set dynamics.
        """
        for joint_name in ['left_inner_finger_pad_joint', 'right_inner_finger_pad_joint']:
            self._p.enableJointForceTorqueSensor(self.robot_id, self.joints[joint_name].id)
            self._p.changeDynamics(self.robot_id, self.joints[joint_name].id, lateralFriction=1)


    def load_robot(self):
        """
        Load robot URDF into the environment.
        """
        urdf_path = os.path.join(self.DEFAULT_PATH, 'meshes/urdf/ur5_robotiq_85.urdf')
        initial_position = [0, 0, 0.9]
        initial_orientation = self._p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = self._p.loadURDF(urdf_path, initial_position, initial_orientation, useFixedBase=True)


    def get_current_tcp(self):
        """
        Get the current Tool Center Point (TCP).
        """
        return self._p.getLinkState(self.robot_id, self.tool_tip_id, physicsClientId=self.client_id)


    def get_current_joint_config(self):
        """
        Get the current joint configuration.
        """
        return [self._p.getJointState(self.robot_id, i)[0] for i in self.arm_joint_indices]


    def get_current_arm_and_gripper_joint_config(self):
        """
        Get the current joint configuration.
        """
        return [self._p.getJointState(self.robot_id, i)[0] for i in self.arm_and_gripper_joint_indices]


    def get_current_arm_world_state(self):
        """
        Get the current world state of the arm.
        """
        return [self._p.getLinkState(self.robot_id, i)[0] for i in self.arm_joint_indices]


    def get_current_gripper_world_state(self):
        """
        Get the current world state of the gripper, left finger, right finger, tool_tip.
        """
        left = self._p.getLinkState(self.robot_id, self.tool_tip_left_id)[0]
        right = self._p.getLinkState(self.robot_id, self.tool_tip_right_id)[0]
        middle = self._p.getLinkState(self.robot_id, self.tool_tip_id)[0]
        return [left, middle, right]

    '''
    ########################################
            Manipulation functions
    ########################################
    '''

    def reset_robot(self, parameters):
        """
        Reset Robot to any configuration.
        parameters: list of 7 values for the arm (6) and gripper (1) joints.
        """
        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            if i == 6:
                self.move_gripper(parameters[i])
                continue
            self._p.resetJointState(self.robot_id, joint.id, parameters[i])
            self._p.setJointMotorControl2(self.robot_id, joint.id, self._p.POSITION_CONTROL,
                                          targetPosition=parameters[i], force=joint.maxForce,
                                          maxVelocity=joint.maxVelocity, positionGain=0.4)
        self.step_simulation(self.per_step_iterations)


    def linear_interpolate_motion_klampt(self, target_pos, target_orientation, verbose=False):
        """
        Interpolates the motion of the robot end effector to the target position and orientation
        using the Klampt library.

        Args:
            target_pos (list): The target position to move to. [x, y, z]
            target_orientation (list): The target orientation to move to. [roll, pitch, yaw]
        """
        # add the initial position to the list of moved positions
        last_moved_positions = [self.get_current_gripper_world_state()]

        current_joint_config = self.get_current_arm_and_gripper_joint_config()
        self.klampt_utils.set_joint_positions(current_joint_config)
        if len(target_orientation) < 4:
            target_ori_quat = self._p.getQuaternionFromEuler(target_orientation)
        else: target_ori_quat = target_orientation

        # add the target point to the pybullet simulation as a sphere
        if verbose:
            vs_id = self._p.createVisualShape(shapeType=self._p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1])
            marker_id = self._p.createMultiBody(baseVisualShapeIndex=vs_id, basePosition=target_pos)

        path = self.klampt_utils.plan_to_cartesian_goal("tool_tip", target_pos, target_ori_quat, num_ik_tries=5, verbose=verbose)

        # remove the target point from the pybullet simulation
        if path is None:
            if verbose:
                self._p.removeBody(marker_id)
            print("!!!!!!!!!!! No path found !!!!!!!!!!!")
            return last_moved_positions
        
        for config in path:
            self.execute_joint_states(config[1:7], absolute=True)
            last_moved_positions.append(self.get_current_gripper_world_state())

        if verbose:
            self._p.removeBody(marker_id)
        return last_moved_positions


    def linear_interpolate_motion_klampt_joint(self, target_joint_config, check_feasibility=True, verbose=False):
        """
        Interpolates the motion of the robot end effector to the target position and orientation
        using the Klampt library.

        Args:
            target_joint_config (list): The target joint configuration to move to. (arm + gripper)
        """
        # add the initial position to the list of moved positions
        last_moved_positions = [self.get_current_gripper_world_state() + self.get_current_arm_world_state()]

        current_joint_config = self.get_current_arm_and_gripper_joint_config()
        self.klampt_utils.set_joint_positions(current_joint_config)

        path = self.klampt_utils.plan_to_joint_goal(target_joint_config, check_feasibility, verbose=verbose)

        if path is None:
            print("!!!!!!!!!!! No path found !!!!!!!!!!!")
            return last_moved_positions
        
        for config in path:
            moved_positions = self.klampt_utils.get_fk_of_links(config, self.klampt_utils.useful_swept_volume_indices)
            last_moved_positions.append(moved_positions)
        
        for config in path:
            self.execute_joint_states(config[1:7], absolute=True)
            if verbose:
                print(f'[linear_interp]   goal:    {config[1:7]}')
                print(f'                  current: {self.get_current_arm_and_gripper_joint_config()[:6]}')

        if verbose:
            # print current joint config
            print("[linear_interp] Current joint config: ", self.get_current_arm_and_gripper_joint_config()[:6])
            # print current tcp
            print("[linear_interp] Current TCP: ", self.get_current_tcp()[0])

        return last_moved_positions


    def linear_interpolate_motion_klampt_joint_traj(self, traj, traj_annotation = None,verbose=False,imagined = False):
        """
        Interpolates the motion of the robot end effector to the target position and orientation
            return: the list of moved positions
        """

        discretized_traj = self.klampt_utils.discretize_trajectory(trajectory = traj,dt = 0.05)
        
        last_moved_positions = []
        if traj is None:
            print("!!!!!!!!!!! No path given !!!!!!!!!!!")
            return last_moved_positions
        if traj_annotation is None:
            traj_annotation = len(traj)*['free']
        for index,config in enumerate(discretized_traj):
            moved_positions = self.klampt_utils.get_fk_of_links(config, self.klampt_utils.useful_swept_volume_indices)
            last_moved_positions.append(moved_positions)
        if(not imagined):
            for config,annot in zip(traj,traj_annotation):
            
                self.execute_joint_states(config[1:7],annotation = annot, absolute=True)
                if verbose:
                    print(f'[linear_interp]   goal:    {config[1:7]}')
                    print(f'                  current: {self.get_current_arm_and_gripper_joint_config()[:6]}')
        
        if verbose:
            print("[linear_interp] Current joint config: ", self.get_current_arm_and_gripper_joint_config()[:6])
            print("[linear_interp] Current TCP: ", self.get_current_tcp()[0])

        return last_moved_positions


    def execute_joint_states(self, target_joint_states, absolute=False, annotation = 'free'):
        '''
        Execute joint states for the robot.
        Args:
            target_joint_states: target joint states to reach
            absolute: simulate until pose reached (True) or only for given number of steps, disregarding the final target pose (False)
            annotation: annotation for the control settings
        '''

        self.control_robot(target_joint_states, annotation=annotation)
        if absolute:
            self.simulate_until_motion_done(target_joint_states)
        self.step_simulation(self.per_step_iterations)


    def control_robot(self, target_joint_states,annotation = 'free',verbose = False):
        '''
        Execute and simulate joint states for the robot.
        Args:
            target_joint_states:  target joint states to reach
            annotation: annotation for the control settings
            verbose: print the annotation
        '''

        position_gain = 0.005 #self.control_settings_dict['position_gain'].get(annotation, 0.005)
        if(verbose):
            print('[UR5 ENV:]annotation : {}'.format(annotation))
        for i, name in enumerate(self.controlJoints[:6]):
            joint = self.joints[name]
            self._p.setJointMotorControl2(self.robot_id, joint.id, self._p.POSITION_CONTROL,
                                          targetPosition=target_joint_states[i], force=joint.maxForce*4,
                                          maxVelocity=joint.maxVelocity*5, positionGain=position_gain)


    def simulate_until_motion_done(self, target_joint_states, max_it=1000):
        '''
        Simulate the robot until it reaches the target joint states to a given threshold.
        Args:
            target_joint_states: target joint states to reach
            max_it: maximum number of iterations to simulate
        '''

        past_joint_pos = deque(maxlen=5)
        joint_state = self._p.getJointStates(self.robot_id, [i for i in range(1,7)])
        joint_pos = list(zip(*joint_state))[0]
        n_it = 0
        while not np.allclose(joint_pos, target_joint_states, atol=1e-2) and n_it < max_it:
            self.step_simulation(self.per_step_iterations)
            n_it += 1
            # Check to see if the arm can't move any close to the desired joint position
            if len(past_joint_pos) == 5 and np.allclose(past_joint_pos[-1], past_joint_pos, atol=1e-2):
                break
            past_joint_pos.append(joint_pos)
            joint_state = self._p.getJointStates(self.robot_id, [i for i in range(1,7)])
            joint_pos = list(zip(*joint_state))[0]


    def get_ik_joints(self, position, orientation, robot_id=None, link=None):
        """
        pybullet internal inverse kinematics solver.
        Args:
            position: position of the end effector
            orientation: orientation of the end effector
            robot_id: robot id; default is self.robot_id
            link: desired link id; default is self.eef_id
        Returns: joint states (6D) for the given position and orientation
        """

        if robot_id is None:
            robot_id = self.robot_id
        if link is None:
            link = self.eef_id
        if len(orientation) == 3:
            orientation = self._p.getQuaternionFromEuler(orientation)

        joints = self._p.calculateInverseKinematics(robot_id, link, position, orientation,
                                                    solver=self._p.IK_DLS, maxNumIterations=1000,
                                                    residualThreshold=1e-5, physicsClientId=self.client_id)
        return list(joints)[:6]


    '''
    ########################################
            Gripper functions
    ########################################
    '''

    def move_gripper(self, gripper_opening_length):
        """
        Move the gripper to a given opening length.
        Args:
            gripper_opening_length: opening length of the gripper 0.0 to 0.085
        """

        gripper_opening_angle = 0.715 - np.arcsin((gripper_opening_length - 0.010) / 0.1143)  # angle calculation
        self.controlGripper(controlMode=self._p.POSITION_CONTROL, targetPosition=gripper_opening_angle)
        #self.simulate_grasp_until_done(gripper_opening_angle)
        self.step_simulation(self.per_step_iterations)
        return


    def simulate_grasp_until_done(self, finger_angle, max_it=1000):
        '''
        Simulate the gripper until it reaches the target joint state to a given threshold.
        Args:
            finger_angle: target joint states to reach
            max_it: maximum number of iterations to simulate
        '''

        finger_joint = self.joints["finger_joint"].id
        finger_joint_state = self._p.getJointState(self.robot_id, finger_joint)[0]

        n_it = 0
        while not np.isclose(finger_joint_state, finger_angle, atol=1e-2) and n_it < max_it:
            self.step_simulation(self.per_step_iterations)
            n_it += 1
            finger_joint_state = self._p.getJointState(self.robot_id, finger_joint)[0]


    def add_constraint_to_gripper(self, obj_id):
        """
        Add a constraint between the gripper and the object to simulate tight grasping
        """
        # get relative positions
        object_world_position, object_world_orientation = self._p.getBasePositionAndOrientation(obj_id)

        # Get the world position and orientation of the gripper link and the object
        gripper_link_state = self._p.getLinkState(self.robot_id, self.grasp_link_id)
        gripper_world_position = gripper_link_state[4]  # Link world position
        gripper_world_orientation = gripper_link_state[5]  # Link world orientation

        # Calculate the position of the object in the local frame of the gripper link
        gripper_inverse_position, gripper_inverse_orientation = self._p.invertTransform(gripper_world_position,
                                                                                       gripper_world_orientation)
        local_position, local_orientation = self._p.multiplyTransforms(gripper_inverse_position,
                                                                      gripper_inverse_orientation,
                                                                      object_world_position, object_world_orientation)

        # Create a fixed constraint between the gripper link and the object
        constraint_id = self._p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.grasp_link_id,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,  # -1 means the base of the object (no link)
            jointType=self._p.JOINT_FIXED,  # Fixed joint
            jointAxis=[0, 0, 0],  # Not used for fixed joint
            parentFramePosition=local_position,  # Local position of the object relative to the gripper
            childFramePosition=[0, 0, 0],  # Attach the object at its center
            parentFrameOrientation=local_orientation,
            childFrameOrientation=[0, 0, 0]
        )
        return constraint_id

    '''
    ########################################
            Collision functions
    ########################################
    '''

    def check_for_contact(self, obj_id, idx):
        ''' check for contact between the object and the robot arm joints
        Parameters:
            obj_id: object id
            idx: index of the object in the current object list
        Returns:
            object_contact: True if the object is in contact with the robot arm joints, False otherwise
        '''
        contact, close = [], []
        for j in self.arm_joint_indices:
            contact, close = self.collision_checks(contact, close, self.robot_id, obj_id, link_A=j)
            self.object_contact[idx] = True if contact or close else False
        return self.object_contact

if __name__ == '__main__':
    env = RobotEnv(render=True, show_vis=False)
    while True:
        env.step_simulation(env.per_step_iterations)
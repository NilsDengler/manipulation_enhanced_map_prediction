#!/usr/bin/python3
from shelf_gym.environments.ur5_environment import RobotEnv
import sys, os
import numpy as np
import gymnasium as gym
from shelf_gym.utils.camera_utils import Camera
from shelf_gym.utils.object_utils import Objects

class ShelfEnv(RobotEnv):
    DEFAULT_ASSET_PATH = os.path.join(os.path.dirname(__file__), '../meshes/urdf/')
    def __init__(self,render=False, shared_memory=False, hz=240, max_obj_num = 20, max_occupancy_threshold=.4, use_ycb=False, show_vis = False):
        super().__init__(render=render, shared_memory=shared_memory, hz=hz, show_vis=show_vis)
        # place floor, shelf and table
        self.default_quaternion = self._p.getQuaternionFromEuler([0, 0, 0])
        self.build_env()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)
        self.max_occupancy_threshold = max_occupancy_threshold
        self.max_obj_num = max_obj_num

        # initialize object class
        use_ycb = use_ycb
        self.obj = Objects(self.client_id, use_ycb, self.max_obj_num)
        self.init_objects(self.max_obj_num, use_ycb)
        self.current_obj_ids = []
        self.current_obj_classes = []
        self.initialize_object_info()

        # initialize camera
        self.hand_camera = Camera(640, 480)

        #step simulation first time
        self.step_simulation(self.per_step_iterations)

        #save initial world state
        self.world_id = self._p.saveState()

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        print("Environment Initialized")
        self.global_steps = 0


    def init_objects(self, max_sample_number=10, use_ycb=False):
        for name in self.obj.obj_urdf_names:
            for _ in range(max_sample_number):
                if use_ycb: path = f'{self.DEFAULT_ASSET_PATH}ycb_objects/{name}/model_textureless.urdf'
                else: path = f'{self.DEFAULT_ASSET_PATH}placing_objects/{name}.urdf'
                pos = [np.random.uniform(2.0, 4.0), np.random.uniform(-2., -4), 0.3]
                self.obj.load_obj(path, pos, 0, mod_stiffness=False, name=name)

        #moving this out of the loop GREATLY accelerates startup but makes it a bit less stable and can cause object drop
        #self.step_simulation(240 / 30)
        #self.obj.wait_until_all_still()


    def initialize_object_info(self):
        '''initialize object information'''
        self.current_obst_pos = np.zeros((len(self.current_obj_ids), 6))
        self.object_tilted = np.zeros((len(self.current_obj_ids)))
        self.object_contact = np.zeros((len(self.current_obj_ids)+1))


    def check_for_tilted(self, obj_id, idx):
        ''' check if the object is tilted
        Parameters:
            obj_id: object id
            idx: index of the object in the current object list
        Returns:
            object_tilted: True if the object is tilted, False otherwise
        '''
        self.object_tilted[idx] = self.obj.check_object_drop(obj_id, reset=False)
        return self.object_tilted


    def build_env(self):
        '''
        Builds the environment by loading the plane, table, shelf and wall
        '''

        self.planeID = self._p.loadURDF('plane.urdf')
        self.UR5Stand_id = self._p.loadURDF(self.DEFAULT_PATH+'meshes/urdf/environment/table.urdf', [0.0, -0.15, 0.45],
                                            self.default_quaternion,
                                            useFixedBase=True)
        self.shelf_id = self._p.loadURDF(self.DEFAULT_PATH+'meshes/urdf/environment/shelf.urdf', [0., .9, 0.],
                                         self._p.getQuaternionFromEuler([0, 0, np.pi]),
                                         useFixedBase=True)
        self.wall_id = self._p.loadURDF(self.DEFAULT_PATH+'meshes/urdf/environment/wall.urdf', [0., 1.13, 0.],
                                         self.default_quaternion,
                                         useFixedBase=True)
        
        # load shelf racks 
        self.rack_ids = []
        self.rack_heights = [2.0, 1.56, 1.18]

        # load shelf racks into the envronment outside the shelf for now
        for i in range(3):
            self.rack_ids.append(self._p.loadURDF(self.DEFAULT_PATH+'meshes/urdf/environment/shelf_rack.urdf', [0., -1.5, 0.05*(i+1)],
                                                  self.default_quaternion, useFixedBase=True))
        self.put_racks_in_shelf()


    def put_racks_in_shelf(self):
        for i in range(3):
            self._p.resetBasePositionAndOrientation(self.rack_ids[i], [0., 0.9, self.rack_heights[i]], self.default_quaternion)


    def remove_racks_from_shelf(self):
        for i in range(3):
            self._p.resetBasePositionAndOrientation(self.rack_ids[i], [0., -1.5, 0.05*(i+1)], self.default_quaternion)


    def initial_reset(self):
        self._p.restoreState(self.world_id)
        self.reset_robot(self.initial_parameters)
        return

    def add_other_elements_to_instance_dict(self,instance_to_class_dict,all_background = True):
        tmp = {}
        num_object_types = len(self.obj.obj_urdf_names)
        
        if(not all_background):
            shelf_class = num_object_types
            rack_class = num_object_types+1
            wall_class = num_object_types+2
            table_class = num_object_types+3
            plane_class = num_object_types + 4
            robot_class = num_object_types + 5
            background_class = num_object_types + 6
        else:
            shelf_class = rack_class = table_class = plane_class = robot_class = wall_class = background_class = num_object_types

        for i in self.rack_ids:
            tmp.update({i:rack_class})
        tmp.update({self.shelf_id:shelf_class})
        tmp.update({self.wall_id:wall_class})
        tmp.update({self.UR5Stand_id:table_class})
        tmp.update({self.planeID:plane_class})
        tmp.update({self.robot_id:robot_class})
        tmp.update({5000:background_class})
        instance_to_class_dict.update(tmp)
        return instance_to_class_dict


if __name__ == '__main__':
    env = ShelfEnv(render=True, show_vis=True, use_ycb=True)
    while True:
        env.step_simulation(env.per_step_iterations)
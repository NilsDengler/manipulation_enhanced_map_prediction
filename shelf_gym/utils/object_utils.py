import os
import time

import pybullet as pb
import numpy as np
from shelf_gym.utils.placement_logic_utils import (
    CircularObjects,
    BoxObjects,
    create_placed_object,
    get_placement_grid,
    get_starting_probabilities,
    get_starting_occupancies,
    get_mk_sum,
    get_original_valid_placing_area,
    grid_to_index,
    sample_point_with_alignment,
    display_arrangements,
)
from shapely.geometry import Point, Polygon
from shapely.affinity import translate
import shapely
import pdb
import seaborn as sns
import yaml
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

SHELF_HEIGHT = 0.80
Z_TABLE_TOP = 0.85

class Objects():
    def __init__(self, client_id, use_ycb=True, max_obj_num=20, physics=True):
        self.object_config_path = os.path.join(os.path.dirname(__file__), 'config/default_object_class_dict.yaml')
        self.client_id = client_id
        self.z_shelf_top = 0.905
        self.z_shelf_height = 0.905
        self.use_ycb = use_ycb
        self.use_minimal = False
        targetable_objects = self.init_object_classes()
        self.init_obj_properties()
        self.obj_colors = np.array(sns.color_palette("husl",len(self.obj_urdf_names)+1))
        self.obj_color_dict = {}
        for i, name in enumerate(self.obj_urdf_names):
            self.obj_color_dict.update({name:self.obj_colors[i]})

        self.placing_algo = ShelfPlacement(0.01, self.obj_class_dict, targetable_objects,  max_samples_per_class=max_obj_num)
        if physics:
            self.cover = self.create_cover(0.001, 0.8, 0.4)
            pb.resetBasePositionAndOrientation(self.cover, [-2, -2, 0.1], pb.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.client_id), physicsClientId=self.client_id)
            pb.changeDynamics(self.cover, -1, mass=-1, physicsClientId=self.client_id)


    def init_object_classes(self):
        if self.use_ycb:
            if self.use_minimal:
                self.obj_urdf_names = ['YcbChipsCan', 'YcbTomatoSoupCan', 'YcbCrackerBox']
                targetable_objects = [1]
            else:
                self.obj_urdf_names = ['YcbTomatoSoupCan', 'Ycbsuger1', 'YcbPottedMeatCan', 'YcbOrionPie',
                                      'YcbMustardBottle', 'YcbMasterChefCan', 'YcbGelatinBox', 'YcbCrackerBox',
                                      'YcbChipsCan', 'YcbBleachCleanser', "backmeel", 'collezione', 'muesli', 'vollmilch']
                targetable_objects = [1, 2, 3, 5]
        else:
            self.obj_urdf_names = ['box_05_05', 'box_06_12', 'box_09_14', 'cylinder_03', 'cylinder_04', 'cylinder_08']
            targetable_objects = [0, 1, 2, 3, 4, 5]
        return targetable_objects


    def load_obj_dict(self, file_path):
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)

        objects = []
        for obj_data in yaml_data['obj_class_dict']:
            if obj_data['name'] in self.obj_urdf_names:
                obj_type = obj_data['obj']['type']
                if obj_type == "CircularObjects":
                    obj_instance = CircularObjects(radius=obj_data['obj']['radius'])
                elif obj_type == "BoxObjects":
                    obj_instance = BoxObjects(height=obj_data['obj']['height'], width=obj_data['obj']['width'])
                else:
                    continue

                objects.append({
                    'obj': obj_instance,
                    'radius_of_influence': obj_data['radius_of_influence'],
                    'affinities': obj_data['affinities'],
                    'self_affinity': obj_data['self_affinity'],
                    'alignment': obj_data['alignment'],
                    'name': obj_data['name'],
                    'is_large': obj_data['is_large']
                })
        return objects


    def init_obj_properties(self):
        self.obj_ids = []
        self.obj_mass = []
        self.obj_names = []
        self.obj_positions = []
        self.obj_orientations = []
        self.use_minimal = False
        self.id_to_class_dict = {}
        self.obj_class_dict = self.load_obj_dict(self.object_config_path)


    def load_obj(self, path, pos, yaw, mod_stiffness=False, name="default"):
        orn = pb.getQuaternionFromEuler([0, 0, yaw], physicsClientId=self.client_id)
        obj_id = pb.loadURDF(path, pos, orn, physicsClientId=self.client_id)
        pb.setCollisionFilterGroupMask(obj_id, -1, 0, 0, physicsClientId=self.client_id)
        # Change dynamics
        mass = pb.getDynamicsInfo(obj_id, -1, physicsClientId=self.client_id)[0]
        dynamics_params = {
            "lateralFriction": 1 if mod_stiffness else 0.5,
            "rollingFriction": 0.001 if mod_stiffness else 0.002,
            "spinningFriction": 0.002 if mod_stiffness else 0.001,
            "restitution": 0.01,
            "mass": 0,
        }

        if mod_stiffness:
            dynamics_params.update({"contactStiffness": 100000, "contactDamping": 0.0})

        pb.changeDynamics(obj_id, -1, physicsClientId=self.client_id, **dynamics_params)

        # Change visual color
        color = self.obj_color_dict[name]
        rgba_color = np.array([color[0], color[1], color[2], 1])
        pb.changeVisualShape(obj_id, -1, rgbaColor=rgba_color, physicsClientId=self.client_id)

        # Store object data
        self.obj_mass.append(mass)
        self.obj_ids.append(obj_id)
        self.obj_names.append(name)
        self.obj_positions.append(pos)
        self.obj_orientations.append(orn)
        return obj_id, pos, orn


    def wait_until_still(self, objID, max_wait_epochs=10):
        for _ in range(max_wait_epochs):
            self.step_simulation(240/30)
            if self.is_still(objID):
                return

    def wait_until_all_still(self, max_wait_epochs=1000):
        for _ in range(max_wait_epochs):
            self.step_simulation(240/30)
            if np.all(list(self.is_still(obj_id) for obj_id in self.obj_ids)):
                return

    def is_still(self, handle):
        still_eps = 1e-2
        lin_vel, ang_vel = pb.getBaseVelocity(handle, physicsClientId=self.client_id)
        pb.resetBaseVelocity(handle, [0, 0, 0], [0, 0, 0], physicsClientId=self.client_id)
        return np.abs(lin_vel).sum() + np.abs(ang_vel).sum() < still_eps


    def update_obj_states(self, current_obj_ids=None):
        if current_obj_ids is None:
            for i, obj_id in enumerate(self.obj_ids):
                pos, orn = pb.getBasePositionAndOrientation(obj_id, physicsClientId=self.client_id)
                self.obj_positions[i] = pos
                self.obj_orientations[i] = orn
                return self.obj_positions, self.obj_orientations
        else:
            current_obj_pos = []
            current_obj_orn = []
            for i, obj_id in enumerate(current_obj_ids):
                pos, orn = pb.getBasePositionAndOrientation(obj_id)
                current_obj_pos.append(pos)
                current_obj_orn.append(orn)
            return current_obj_pos, current_obj_orn





    def reset_obj(self, obj_id, r_x, r_y, yaw=None):
        yaw = np.random.uniform(-np.pi / 2, np.pi / 2) if yaw is None else yaw
        pos = np.array([r_x, r_y, self.z_shelf_height])

        orn = pb.getQuaternionFromEuler([0, 0, yaw], physicsClientId=self.client_id)
        aabb = pb.getAABB(obj_id, -1, physicsClientId=self.client_id)
        pos[2] += (aabb[1][2] - aabb[0][2]) / 2

        pb.resetBasePositionAndOrientation(obj_id, pos.tolist(), orn, physicsClientId=self.client_id)
        self.wait_until_still(obj_id, 1)
        self.update_obj_states()
        return True


    def check_object_drop(self, obj_id, reset=False, name=""):
        pos, quat_orn = pb.getBasePositionAndOrientation(obj_id, physicsClientId=self.client_id)
        orn = np.array(pb.getEulerFromQuaternion(quat_orn, physicsClientId=self.client_id))

        pos = np.array(pos)
        bounds = np.array([0.8, 1.3])
        if reset:
            if name=="Ycbsuger3":
                return False
            return np.logical_or(pos[2] < bounds[0], pos[2] > bounds[1])
            #return np.any(np.abs(orn[:2]) > 0.01) or np.logical_or(pos[2] < bounds[0], pos[2] > bounds[1])
        return np.logical_or(pos[2] < bounds[0], pos[2] > bounds[1])

        #return np.any(np.abs(orn[:2]) > np.pi / 4) or np.logical_or(pos[2] < bounds[0], pos[2] > bounds[1])


    def check_outside_shelf(self, obj_id):
        pos, quat_orn = pb.getBasePositionAndOrientation(obj_id, physicsClientId=self.client_id)
        if pos[2] < 0.7:
            return True
        else: False


    def check_all_objects_inside_shelf(self,obj_ids):
        return all([not self.check_outside_shelf(id) for id in obj_ids])


    def check_all_object_drop(self, obj_ids):
        return any([self.check_object_drop(id) for id in obj_ids])


    def get_object_indx(self, obj_type, obj_list):
        ids = []
        idx = -1
        while True:
            try:
                idx = obj_list.index(obj_type, idx + 1)
                ids.append(idx)
            except ValueError:
                break
        return ids


    def sample_objects_on_shelf(self, use_occupancy, occupancy_threshold=0.4, sample_object_num=12, alignment=0, sample_hard=True):
        placed_objects = self.placing_algo.draw_arrangement(alignment, use_occupancy=use_occupancy, occupancy_threshold=occupancy_threshold, total_objects=sample_object_num,  sample_hard= sample_hard)
        return placed_objects


    def reset_dynamics(self, ids):
        for i, obj_id in enumerate(ids):
            pb.changeDynamics(obj_id, -1, physicsClientId=self.client_id, mass=0)
            pb.setCollisionFilterGroupMask(obj_id, -1, 0, 0, physicsClientId=self.client_id)


    def physically_place_objects(self, placed_objects):
        object_name_ids = []
        object_ids = []
        object_classes = [14]
        object_counter = [0]*len(self.obj_urdf_names)
        pb.resetBasePositionAndOrientation(self.cover, [0., 0.7, 1.1], pb.getQuaternionFromEuler([0, 0, np.pi/ 2]), physicsClientId=self.client_id)

        object_name_ids = [self.get_object_indx(n, self.obj_names) for n in self.obj_urdf_names]

        for o in placed_objects:
            name_idx = self.obj_urdf_names.index(o.name)
            obj_ids = object_name_ids[name_idx]
            pb.setCollisionFilterGroupMask(self.obj_ids[obj_ids[object_counter[name_idx]]], -1, 1, 1, physicsClientId=self.client_id)
            pb.changeDynamics(self.obj_ids[obj_ids[object_counter[name_idx]]], -1, physicsClientId=self.client_id, mass=self.obj_mass[obj_ids[object_counter[name_idx]]])

            object_ids.append(self.obj_ids[obj_ids[object_counter[name_idx]]])
            object_classes.append(o.object_class)
            self.reset_obj(self.obj_ids[obj_ids[object_counter[name_idx]]], o.shape.centroid.x, o.shape.centroid.y, np.deg2rad(o.angle))
            object_counter[name_idx] += 1
            self.id_to_class_dict.update({object_ids[-1]:o.object_class})
        self.update_obj_states()
        self.wait_until_all_still(max_wait_epochs=5)
        pb.resetBasePositionAndOrientation(self.cover, [3, 0.75, 0.9], pb.getQuaternionFromEuler([0, 0, np.pi/ 2]), physicsClientId=self.client_id)
        return object_ids, object_classes


    def step_simulation(self, num_steps):
        """Perform a simulation step.
                Args:
                - num_steps: Number of steps to simulate.
                """
        for _ in range(int(num_steps)):
            pb.stepSimulation(physicsClientId=self.client_id)
            pb.performCollisionDetection()


    def get_id_to_class_dict(self):
        return self.id_to_class_dict


    def get_obj_ids(self):
        return self.obj_ids


    def create_cover(self, w, l, h, mass=-1, color=(1, 0, 0, 1), **kwargs):
        '''Create a cover in front of shelf to prevent objects from falling off'''

        pose = ((0., 0., 0.), pb.getQuaternionFromEuler([0, 0, 0],  physicsClientId=self.client_id))
        box_geom = {'shapeType': pb.GEOM_BOX, 'halfExtents': [w / 2., l / 2., h / 2.]}
        collision_args = {
            'collisionFramePosition': pose[0],
            'collisionFrameOrientation': pose[1],
            'physicsClientId': self.client_id,
        }
        collision_args.update(box_geom)
        if 'length' in collision_args:
            # pybullet bug visual => length, collision => height
            collision_args['height'] = collision_args['length']
            del collision_args['length']

        visual_args = {
            'rgbaColor': color,
            'visualFramePosition': pose[0],
            'visualFrameOrientation': pose[1],
            'physicsClientId': self.client_id,
        }
        visual_args.update(box_geom)

        visual_id = pb.createVisualShape(**visual_args)
        collision_id = pb.createCollisionShape(**collision_args)

        return pb.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id,
                                  baseVisualShapeIndex=visual_id, physicsClientId=self.client_id)



class ShelfPlacement:
    def __init__(self, resolution, placeable_objects, targetable_objects,
                 max_retries=5, workspace=Polygon([(-0.38, 0.7), (-0.38, 1.08), (0.38, 1.08), (0.38, 0.7)]),
                 search_area=Polygon([(-0.38, 0.85), (-0.38, 1.08), (0.38, 1.08), (0.38, 0.85)]),  max_samples_per_class=25):
        """_summary_

        Args:
            resolution (float): resolution with which to discretize the workspace
            placeable_objects (dict): dictionary containing the placeable objects, radius of influence, affinities and its self affinity
            targetable_objects (list): list of all object classes that can be targets of search (preferrably smaller objects to avoid trivial search problems)
            max_retries (int, optional): Number of sampling retries after a placement attempt fails (i.e. resampling different placement angles). Defaults to 5.
            workspace (Shapely.geometry.Polygon, optional): Workspace where we can place objects in the shelf. Defaults to Polygon([(-0.38, 0.7), (-0.38, 1.08), (0.38, 1.08), (0.38, 0.7)]).
            search_area (Shapely.geometry.Polygon, optional): part of the shelf which should result in non-trivial search problems. Defaults to Polygon([(-0.38, 0.85), (-0.38, 1.08), (0.38, 1.08), (0.38, 0.85)]).
        """
        self.resolution = resolution
        self.placeable_objects = placeable_objects
        self.max_retries = max_retries
        self.targetable_objects = targetable_objects
        self.workspace = workspace
        self.n_classes = len(self.placeable_objects)
        self.search_area = search_area
        self.max_samples_per_class = max_samples_per_class

    def adjust_conditional_probabilities(self, current_probs, placed_objects, all_valid_points):
        for placed_object in placed_objects:
            radius_of_influence = placed_object.get_radius_of_influence_shape()
            affected_points = all_valid_points.intersection(radius_of_influence)
            prob_vec = placed_object.conditional_prob
            coords = []
            if (not affected_points.is_empty):
                if (affected_points.geom_type != 'Point'):
                    for pt in affected_points.geoms:
                        coords.append(pt.xy)
                else:
                    coords.append(affected_points.xy)
                coords = np.array(coords).reshape(-1, 2)
                indices = grid_to_index(coords, self.workspace, self.resolution)
                x_index = indices[:, 0]
                y_index = indices[:, 1]
                current_probs[x_index, y_index, :] *= prob_vec
                current_probs[x_index, y_index, :] = current_probs[x_index, y_index, :] / current_probs[x_index, y_index, :].sum(axis=1, keepdims=True)
        return current_probs


    def draw_arrangement(self, alignment, use_occupancy=True, occupancy_threshold=0.40, total_objects=12,  max_angle=180, search=True,  sample_hard=True):
        """_summary_

        Args:
            alignment (float): A parameter indicating the alignment bias in this arrangement in [0,1] where 0 is random and 1 is strongly ordered rows of identical objects
            occupancy_threshold (int): Total number of objects to be placed in this arrangement
            max_angle (float, optional): Maximum angle that objects can be placed relative to their canonical orientation. Defaults to 45.
            search (bool, optional): Whether this arrangement is for a search problem or not. If true, it samples a class to be searched and ensures that only one instance of that class is placed. Defaults to True.

        Returns:
            list[PlacedObjects]: A list of all PlacedObjects in the scene, containing their shapes and classes
        """
        # we first update
        for placeable_object in self.placeable_objects:
            placeable_object.update({'alignment': alignment})
        candidate_points, x_span, y_span, mg = get_placement_grid(self.workspace, self.resolution)
        starting_probs = get_starting_probabilities(x_span, y_span, self.n_classes)
        retries = 0
        placed_objects = []
        placed_area = []
        placement_class_counter = np.zeros(len(self.placeable_objects))
        workspace_area = self.workspace.area
        om = get_starting_occupancies(x_span, y_span).transpose()
        # we begin by choosing an object to be the search target, if this is a search problem
        if (search):
            angle = max_angle * np.random.rand()
            target_object_class = np.random.choice(self.targetable_objects, 1)[0]
            name = self.placeable_objects[target_object_class]['name']
            candidate_target_object = self.placeable_objects[target_object_class]['obj'].get_generic_shape(angle)
            valid_placing_points = \
            self.obtain_valid_placing_region(self.search_area, [candidate_target_object], [self.search_area],
                                        candidate_points)[0]
            selected_placement = np.random.choice(np.arange(len(valid_placing_points.geoms)), 1)[0]
            chosen_point = valid_placing_points.geoms[selected_placement]
            chosen_coords = np.array(chosen_point.coords)
            placed_shape = self.placeable_objects[target_object_class]['obj'].place(chosen_coords[0, 0],
                                                                                    chosen_coords[0, 1], angle)
            placed_object = create_placed_object(placed_shape, self.placeable_objects[target_object_class],
                                                 target_object_class, self.n_classes, angle, name)
            placed_objects.append(placed_object)
            placed_area.append(placed_object.get_area())
            placement_class_counter[target_object_class] += 1
            self.placeable_objects[target_object_class]['obj'].reset_dimensions()

            display_arrangements(placed_objects, 10)
        else:
            # if it is not a search problem, set target_object_class to -1
            target_object_class = - 1
        occupany_ratio = sum(placed_area)/workspace_area
        if use_occupancy:
            t0 = time.time()
            count = 0
            while (occupany_ratio < occupancy_threshold):
                count += 1
                t1 = time.time()
                placed_objects, placed_area, placement_class_counter, occupany_ratio, retries, break_criteria = \
                    self.object_positions_sampling_loop_parallel(max_angle, placed_objects, starting_probs,  candidate_points,
                                                        om, placed_area, placement_class_counter, workspace_area,
                                                        target_object_class, retries,  sample_hard=sample_hard)
                #print("in loop", time.time() - t1)
                if break_criteria:
                    break
            #print("full loop:", time.time() - t0)
            #print("count:", count)
        else:
            while (len(placed_objects) < total_objects):
                placed_objects, placed_area, placement_class_counter, occupany_ratio, retries, break_criteria = \
                    self.object_positions_sampling_loop_parallel(max_angle, placed_objects, starting_probs,  candidate_points,
                                                        om, placed_area, placement_class_counter, workspace_area,
                                                        target_object_class, retries, sample_hard=sample_hard)
                if break_criteria:
                    break

        return placed_objects


    def object_positions_sampling_loop(self, max_angle, placed_objects, starting_probs, candidate_points, om, placed_area,
                                       placement_class_counter, workspace_area, target_object_class, retries=0):
        t0 = time.time()
        angle = max_angle * np.random.rand()
        occupany_ratio = sum(placed_area) / workspace_area
        # obtain placeable objects
        # then we get representative sample objects:
        t1 = time.time()
        testing_shapes = [i['obj'].get_generic_shape(angle) for i in self.placeable_objects]
        print("shape loop: ", time.time() - t1)

        # now, we get a per-shape valid free workspace:
        t2 = time.time()
        valid_free_space = self.obtain_valid_free_space(self.workspace, placed_objects, testing_shapes,
                                                        target_object_class)
        print("obtain free space: ", time.time() - t2)

        print("first half half: ", time.time() - t0)
        # we can then intersect this with the original valid workspace:
        t3 = time.time()
        valid_placing_points = self.obtain_valid_placing_region(self.workspace, testing_shapes, valid_free_space,
                                                                candidate_points)
        print("obtain_valid_placing_region: ", time.time() - t3)
        t4 = time.time()

        all_valid_points = shapely.unary_union(valid_placing_points)
        print("unary_union: ", time.time() - t4)
        t5 = time.time()

        # we then set the points probabilities of the invalid points for each geometry to be equal to zero:
        current_probs = self.adjust_probabilities_by_occupancy(starting_probs, valid_placing_points, self.n_classes)
        print("adjust_probabilities_by_occupancy: ", time.time() - t5)

        # finally, before sampling, we add the pairwise effects of the placed objects before sampling
        t6 = time.time()
        if (len(placed_objects) > 0):
            current_probs = self.adjust_conditional_probabilities(current_probs, placed_objects, all_valid_points)
        print("adjust_conditional_probabilities: ", time.time() - t6)

        print("first half: ", time.time() - t0)
        t1 = time.time()
        max_class_range_indx = np.where(placement_class_counter > self.max_samples_per_class - 1)
        current_probs[:, :, max_class_range_indx] = 0
        current_probs[:, :, max_class_range_indx]
        current_probs = current_probs / current_probs.sum(axis=-1, keepdims=True)
        current_probs = np.nan_to_num(current_probs, 1 / self.n_classes)
        if np.any(np.logical_not(np.isclose(current_probs.sum(axis=-1), 1, atol=0.00001))):
            pdb.set_trace()
        # we can now sample from the valid points a point for placement:
        if (not all_valid_points.is_empty):
            if (all_valid_points.geom_type != 'Point'):
                chosen_point, _ = sample_point_with_alignment(om.copy(), all_valid_points, self.workspace,
                                                              self.resolution, placed_objects, scale=100)

            else:
                chosen_point = all_valid_points
            chosen_coords = np.array(chosen_point.coords)
            chosen_index = grid_to_index(chosen_coords, self.workspace, self.resolution).reshape(2)
            generative_probs = current_probs[chosen_index[0], chosen_index[1], :]
            if (np.any(np.isnan(generative_probs))):
                retries += 1
                print('failed to place, retries = {}'.format(retries))
                if (retries > self.max_retries):
                    print('concluding with fewer shapes than desired, unfeasible arrangement')
                    return placed_objects, placed_area, placement_class_counter, occupany_ratio, retries, True
            else:
                chosen_object = np.random.choice(np.arange(self.n_classes).astype(int), 1, p=generative_probs)[0]

                # finally, we place the object in the scene
                placed_shape = self.placeable_objects[chosen_object]['obj'].place(chosen_coords[0, 0],
                                                                                  chosen_coords[0, 1], angle)
                name = self.placeable_objects[chosen_object]['name']
                placed_object = create_placed_object(placed_shape, self.placeable_objects[chosen_object], chosen_object,
                                                     self.n_classes, angle, name)
                placed_objects.append(placed_object)
                placed_area.append(placed_object.get_area())
                placement_class_counter[chosen_object] += 1
                occupany_ratio = sum(placed_area) / workspace_area

                retries = 0
                self.placeable_objects[target_object_class]['obj'].reset_dimensions()
        else:
            retries += 1
            print('failed to place, retries = {}'.format(retries))
            if (retries > self.max_retries):
                print('concluding with fewer shapes than desired, unfeasible arrangement')
                return placed_objects, placed_area, placement_class_counter, occupany_ratio, retries, True
        print("second half: ", time.time() - t1)

        return placed_objects, placed_area, placement_class_counter, occupany_ratio, retries, False


    def adjust_probabilities_by_occupancy(self, starting_probs, valid_placing_points, n_classes):
        all_valid_points = shapely.unary_union(valid_placing_points)
        current_probs = starting_probs.copy()
        exclusive_points = [all_valid_points.difference(placements)  for placements in valid_placing_points]

        for i, pts in enumerate(exclusive_points):
            if pts.is_empty:
                continue

            if pts.geom_type == 'Point':
                coords = np.array(pts.xy).reshape(1, 2)
            else:
                coords = np.array([pt.xy for pt in pts.geoms]).reshape(-1, 2)

            indices = grid_to_index(coords, self.workspace, self.resolution)
            x_index, y_index = indices[:, 0], indices[:, 1]

            prob_vec = np.full(n_classes, 1 / (n_classes - 1))
            prob_vec[i] = 0

            current_probs[x_index, y_index, :] *= prob_vec
            current_probs[x_index, y_index, :] /= current_probs[x_index, y_index, :].sum(axis=1, keepdims=True)

        return current_probs

    def obtain_valid_free_space(self, workspace, placed_objects, testing_shapes, target_class):
        if not placed_objects:
            return [workspace if idx != target_class else Point(100, 100) for idx in range(len(testing_shapes))]

        placed_shapes = [obj.shape for obj in placed_objects]
        placed_centroids = np.array([shape.centroid.coords[0] for shape in placed_shapes])

        # Prebuild spatial index for faster spatial queries
        placed_tree = shapely.strtree.STRtree(placed_shapes)

        def compute_free_space(idx, obj):
            if idx == target_class:
                return Point(100, 100)

            obj_centroid = np.array(obj.centroid.coords[0])
            potential_intersections = placed_tree.query(obj)

            mk_sums = []
            for placed_shape in potential_intersections:
                tmp = get_mk_sum(obj, placed_shape)
                diff = placed_shape.centroid.coords[0] - tmp.centroid.coords[0]
                tmp_translated = translate(tmp, diff[0], diff[1])
                mk_sums.append(tmp_translated)

            if not mk_sums:
                return workspace

            mk_union = shapely.unary_union(mk_sums)
            return workspace.difference(mk_union)

        valid_free_space = Parallel(n_jobs=-1, backend='threading')(
            delayed(compute_free_space)(idx, obj) for idx, obj in enumerate(testing_shapes)
        )

        return valid_free_space


    # def obtain_valid_free_space(self, workspace, placed_objects, testing_shapes, target_class):
    #     valid_free_space = []
    #     for idx, obj in enumerate(testing_shapes):
    #         mk_sums = []
    #         if (idx != target_class):
    #             if (len(placed_objects) > 0):
    #                 for placed_object in placed_objects:
    #                     tmp = get_mk_sum(obj, placed_object.shape)
    #                     diff = (np.asarray(placed_object.shape.centroid.coords) - np.asarray(tmp.centroid.coords))[0]
    #                     tmp2 = translate(tmp, diff[0], diff[1])
    #                     mk_sums.append(tmp2)
    #                 valid_workspace = workspace
    #                 for mk_sum in mk_sums:
    #                     valid_workspace = valid_workspace.difference(mk_sum)
    #                 valid_free_space.append(valid_workspace)
    #             else:
    #                 valid_free_space.append(workspace)
    #         else:
    #             valid_free_space.append(Point(100, 100))
    #     return valid_free_space


    def obtain_valid_placing_region(self, workspace, testing_shapes, valid_free_space, candidate_points):
        valid_placing_points = []
        for i, free_space in zip(testing_shapes, valid_free_space):
            valid_placing_regions = get_original_valid_placing_area(workspace, i)
            valid_placing_regions = free_space.intersection(valid_placing_regions)
            valid_placing_points.append(candidate_points.intersection(valid_placing_regions))
        return valid_placing_points

    def object_positions_sampling_loop_parallel(self, max_angle, placed_objects, starting_probs, candidate_points, om,
                                                placed_area, placement_class_counter, workspace_area,
                                                target_object_class, retries=0,  sample_hard=True):
        angle = max_angle * np.random.rand()
        occupancy_ratio = sum(placed_area) / workspace_area

        # Prepare testing shapes in parallel
        with ThreadPoolExecutor() as executor:
            testing_shapes = list(executor.map(lambda obj: obj['obj'].get_generic_shape(angle), self.placeable_objects))

        # Valid free space & placement regions
        valid_free_space = self.obtain_valid_free_space_parallel(self.workspace, placed_objects, testing_shapes,
                                                                 target_object_class)
        valid_placing_points = self.obtain_valid_placing_region_parallel(self.workspace, testing_shapes,
                                                                         valid_free_space, candidate_points)
        all_valid_points = shapely.unary_union(valid_placing_points)

        # Adjust placement probabilities
        current_probs = self.adjust_probabilities_by_occupancy(starting_probs, valid_placing_points, self.n_classes)
        if placed_objects:
            current_probs = self.adjust_conditional_probabilities(current_probs, placed_objects, all_valid_points)

        # Exclude over-sampled classes
        current_probs[:, :, placement_class_counter > self.max_samples_per_class - 1] = 0
        current_probs /= current_probs.sum(axis=-1, keepdims=True)
        current_probs = np.nan_to_num(current_probs, nan=1 / self.n_classes)

        # If there's any place to sample from
        if not all_valid_points.is_empty:
            chosen_point = all_valid_points if all_valid_points.geom_type == 'Point' else \
                sample_point_with_alignment(om.copy(), all_valid_points, self.workspace, self.resolution,
                                            placed_objects, scale=100)[0]

            chosen_coords = np.array(chosen_point.coords)
            chosen_index = grid_to_index(chosen_coords, self.workspace, self.resolution).reshape(2)
            generative_probs = current_probs[chosen_index[0], chosen_index[1], :]

            if sample_hard:
                # -------------------
                # 📌 Size & Position Bias
                y_pos = chosen_coords[0, 1]
                y_front, y_back = 0.90, 0.90  # adjust for your shelf
                size_threshold = 0.008  # tune to match median object area

                for idx, obj_dict in enumerate(self.placeable_objects):
                    is_large = obj_dict["is_large"]

                    if y_pos < y_front and is_large:
                        generative_probs[idx] = 1.

                    if y_pos < y_front and not is_large:
                        generative_probs[idx] = 0.

                    if y_pos > y_back and is_large:
                        generative_probs[idx] = 0

                    if y_pos > y_back and not is_large:
                        generative_probs[idx] = 1.
                    # Penalize large objects in the back
                    #if y_pos > y_back and obj_size > size_threshold:
                    #    generative_probs[idx] *= 0.01

            # Normalize again
            if generative_probs.sum() == 0:
                retries += 1
                return placed_objects, placed_area, placement_class_counter, occupancy_ratio, retries, False

            generative_probs /= generative_probs.sum()
            # Sample object
            chosen_object = np.random.choice(np.arange(self.n_classes).astype(int), p=generative_probs)

            # Place it
            placed_shape = self.placeable_objects[chosen_object]['obj'].place(chosen_coords[0, 0], chosen_coords[0, 1],
                                                                              angle)
            name = self.placeable_objects[chosen_object]['name']
            placed_object = create_placed_object(placed_shape, self.placeable_objects[chosen_object], chosen_object,
                                                 self.n_classes, angle, name)
            placed_objects.append(placed_object)
            placed_area.append(placed_object.get_area())
            placement_class_counter[chosen_object] += 1
            occupancy_ratio = sum(placed_area) / workspace_area

            retries = 0
            self.placeable_objects[target_object_class]['obj'].reset_dimensions()
        else:
            retries += 1
            if retries > self.max_retries:
                return placed_objects, placed_area, placement_class_counter, occupancy_ratio, retries, True

        return placed_objects, placed_area, placement_class_counter, occupancy_ratio, retries, False

    # def object_positions_sampling_loop_parallel(self, max_angle, placed_objects, starting_probs, candidate_points, om,
    #                                             placed_area, placement_class_counter, workspace_area,
    #                                             target_object_class,
    #                                             retries=0):
    #     angle = max_angle * np.random.rand()
    #     occupancy_ratio = sum(placed_area) / workspace_area
    #
    #     # Prepare testing shapes in parallel
    #     with ThreadPoolExecutor() as executor:
    #         testing_shapes = list(executor.map(lambda obj: obj['obj'].get_generic_shape(angle), self.placeable_objects))
    #
    #     # Calculate valid free spaces in parallel
    #     valid_free_space = self.obtain_valid_free_space_parallel(self.workspace, placed_objects, testing_shapes,
    #                                                              target_object_class)
    #
    #     # Obtain valid placement points in parallel
    #     valid_placing_points = self.obtain_valid_placing_region_parallel(self.workspace, testing_shapes,
    #                                                                      valid_free_space,
    #                                                                      candidate_points)
    #
    #     all_valid_points = shapely.unary_union(valid_placing_points)
    #
    #     # Adjust probabilities based on occupancy
    #     current_probs = self.adjust_probabilities_by_occupancy(starting_probs, valid_placing_points, self.n_classes)
    #
    #     if len(placed_objects) > 0:
    #         current_probs = self.adjust_conditional_probabilities(current_probs, placed_objects, all_valid_points)
    #
    #     current_probs[:, :, placement_class_counter > self.max_samples_per_class - 1] = 0
    #     current_probs /= current_probs.sum(axis=-1, keepdims=True)
    #     current_probs = np.nan_to_num(current_probs, nan=1 / self.n_classes)
    #
    #     if not all_valid_points.is_empty:
    #         if all_valid_points.geom_type != 'Point':
    #             chosen_point, _ = sample_point_with_alignment(om.copy(), all_valid_points, self.workspace,
    #                                                           self.resolution, placed_objects, scale=100)
    #         else:
    #             chosen_point = all_valid_points
    #
    #         chosen_coords = np.array(chosen_point.coords)
    #         chosen_index = grid_to_index(chosen_coords, self.workspace, self.resolution).reshape(2)
    #         generative_probs = current_probs[chosen_index[0], chosen_index[1], :]
    #
    #         if np.any(np.isnan(generative_probs)):
    #             retries += 1
    #             if retries > self.max_retries:
    #                 return placed_objects, placed_area, placement_class_counter, occupancy_ratio, retries, True
    #         else:
    #             chosen_object = np.random.choice(np.arange(self.n_classes).astype(int), p=generative_probs)
    #             placed_shape = self.placeable_objects[chosen_object]['obj'].place(chosen_coords[0, 0],
    #                                                                               chosen_coords[0, 1], angle)
    #             name = self.placeable_objects[chosen_object]['name']
    #             placed_object = create_placed_object(placed_shape, self.placeable_objects[chosen_object],
    #                                                  chosen_object, self.n_classes, angle, name)
    #             placed_objects.append(placed_object)
    #             placed_area.append(placed_object.get_area())
    #             placement_class_counter[chosen_object] += 1
    #
    #             occupancy_ratio = sum(placed_area) / workspace_area
    #             retries = 0
    #             self.placeable_objects[target_object_class]['obj'].reset_dimensions()
    #     else:
    #         retries += 1
    #         if retries > self.max_retries:
    #             return placed_objects, placed_area, placement_class_counter, occupancy_ratio, retries, True
    #
    #     return placed_objects, placed_area, placement_class_counter, occupancy_ratio, retries, False

    # Parallelized sub-functions
    def obtain_valid_free_space_parallel(self, workspace, placed_objects, testing_shapes, target_class):
        def compute_valid_space(args):
            idx, obj = args
            mk_sums = []
            if idx != target_class and placed_objects:
                for placed_object in placed_objects:
                    tmp = get_mk_sum(obj, placed_object.shape)
                    diff = np.asarray(placed_object.shape.centroid.coords) - np.asarray(tmp.centroid.coords)
                    mk_sums.append(shapely.affinity.translate(tmp, diff[0, 0], diff[0, 1]))
                valid_workspace = workspace
                for mk_sum in mk_sums:
                    valid_workspace = valid_workspace.difference(mk_sum)
                return valid_workspace
            return Point(100, 100) if idx == target_class else workspace

        with ThreadPoolExecutor() as executor:
            valid_free_space = list(executor.map(compute_valid_space, enumerate(testing_shapes)))
        return valid_free_space

    def obtain_valid_placing_region_parallel(self, workspace, testing_shapes, valid_free_space, candidate_points):
        def compute_placing_region(args):
            shape, free_space = args
            valid_region = get_original_valid_placing_area(workspace, shape).intersection(free_space)
            return candidate_points.intersection(valid_region)

        with ThreadPoolExecutor() as executor:
            valid_placing_points = list(executor.map(compute_placing_region, zip(testing_shapes, valid_free_space)))

        return valid_placing_points
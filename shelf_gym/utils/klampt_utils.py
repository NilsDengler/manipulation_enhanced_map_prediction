import copy
import os
import pdb
import time
from klampt import WorldModel
from klampt import vis
from klampt.plan.robotcspace import RobotCSpace
from klampt.plan import cspace
from klampt.plan.robotplanning import plan_to_config
from klampt.model import collide
from klampt.model.trajectory import RobotTrajectory
from klampt.robotsim import TriangleMesh
from klampt import IKSolver
from klampt.model import ik
from typing import List, Tuple, Union
from klampt.model.typing import Config
from klampt.math import so3
from klampt import Geometry3D
from klampt.model import coordinates
import klampt
import numpy as np
import mcubes
import pyfqmr
import logging
import trimesh
from scipy.spatial.transform import Rotation as R


loggers = {}
for package in ['trimesh','pyfqmr','klampt']:
    loggers.update({package:logging.getLogger(package)})
    loggers[package].setLevel(logging.ERROR)

DEF_PATH = os.path.join(os.path.dirname(__file__), "../")

vis.init("PyQt5")
#vis.init("GLUT")

# Default motion planner settings
# ACTIVE_DOFS = 'all'   #this will plan for all DOFs
ACTIVE_DOFS = "auto"  # this will restrict planning to only the DOFs that move
# set up a settings dictionary here.  This is a random-restart + shortcutting
# SBL planner.
PLANNER_SETTINGS = {
    "type": "sbl",
    "perturbationRadius": 0.25,
    "bidirectional": 1,
    "shortcut": 1,
    "restart": 1,
    "restartTermCond": "{foundSolution:1,maxIters:1000}",
}

PLANNER_SETTINGS_RRT = {
    "type": "rrt",
    "perturbationRadius": 0.25,
    "bidirectional": True,
    "shortcut": True,
    "restart": True,
    "restartTermCond": "{foundSolution:1,maxIters:1000}",
}

IS_PLANNER_OPTIMIZING = True
# You might edit the values 500 and 10 to play with how many iterations /time
# to give the planner.
MAX_PLANNER_ITERS = 500
MAX_PLANNER_TIME = 10.0
EDGE_CHECK_RESOLUTION = 0.005

# Setup settings: feel free to edit these to see how the results change
# Simplify geometry or not?
# SIMPLIFY_TYPE = None
# SIMPLIFY_TYPE = 'aabb'
SIMPLIFY_TYPE = "ConvexHull"
DEBUG_SIMPLIFY = 0
# Create CSpace manually
MANUAL_SPACE_CREATION = 0
# Impose a closed-loop constraint on the problem
CLOSED_LOOP_TEST = 0
# Plan to cartesian goals rather than configurations exactly
PLAN_CARTESIAN_TEST = 0
# Show the smoothed path rather than the raw one
SHOW_SMOOTHED_PATH = 1


class KlamptUtils:
    def __init__(self, default_path=DEF_PATH, initial_parameters=None, show_vis=False):

        self.world = WorldModel(f"{default_path}meshes/klampt/klampt_world.xml")
        self.robot = self.world.robot("ur5")
        self.tool_tip = self.robot.link("tool_tip")
        self.show_vis = show_vis
        if (self.show_vis):
            vis.add("world", self.world)
            vis.show()

        self.default_initial_parameters = [1.1297859669433465, -1.4717861879060314, 2.540078550839643,
                                           -1.0682749163925591, 1.130606603250641, 0.0015903135007540987,
                                           0.8000000000000003, 0.0, -0.8025568053451891, 0.0, 0.8025877935073286,
                                           0.8030522033784521, 0.0, -0.8026295430606621, 0.0, 0.8026183714702051]

        initial_parameters = initial_parameters or self.default_initial_parameters
        self.wc = collide.WorldCollider(self.world)

        self.arm_indices = [1, 2, 3, 4, 5, 6]
        self.gripper_indices = [14, 15, 21]
        self.useful_swept_volume_indices = [4, 5, 6, 7, 11, 17, 22]
        self.useful_swept_volume_names = ["wrist_1_link", "wrist_2_link", "wrist_3_link", "ee_link",
                                          "left_outer_finger", "right_outer_finger", "tool_tip"]

        self.pb_to_klampt_gripper_joint_map = {
            6: 10,
            7: 11,
            8: 12,
            9: 13,
            10: 14,
            11: 16,
            12: 17,
            13: 18,
            14: 19,
            15: 15,
        }

        print("[Klamp't] Initialize robot")
        self.init_robot(initial_parameters)
        self.space = RobotCSpace(self.robot, self.wc)
        self.space.eps = EDGE_CHECK_RESOLUTION
        print("[Klamp't] Robot initialized")

        if not (self.check_config_feasibility(self.robot.getConfig())):
            print("[init_robot] Initial configuration is infeasible")

        self.tm_rb = None
        self.tm_rb_name = None

    def init_robot(self, initial_parameters=None):
        """
        Initializes the robot by handling collision settings and setting joint limits.
        """
        # Identify and ignore initial collisions
        self.ignore_collisions(robot_link_test=True)

        # Explicitly ignore collision between specific gripper pads
        #left_pad = self.robot.link("left_inner_finger_pad")
        #right_pad = self.robot.link("right_inner_finger_pad")
        #self.wc.ignoreCollision((left_pad, right_pad))
        #self.robot.enableSelfCollision(left_pad.getIndex(), right_pad.getIndex(), False)

        # Print remaining collisions
        for obj1, obj2 in self.wc.collisionTests():
            if obj1[1].collides(obj2[1]):
                print(f" - Object {obj1[0].getName()} collides with {obj2[0].getName()}")

        if self.robot.selfCollides():
            print("[init_robot] Robot self-collision detected")

        # Adjust joint limits for the gripper (assuming joints 8 to end are for the gripper)
        joint_limits = self.robot.getJointLimits()
        joint_limits[0][8:] = [-float("inf")] * len(joint_limits[0][8:])  # No lower limits
        joint_limits[1][8:] = [float("inf")] * len(joint_limits[0][8:])  # No upper limits
        self.robot.setJointLimits(qmin=joint_limits[0], qmax=joint_limits[1])

        # Apply initial parameters if provided
        if initial_parameters is not None:
            self.set_joint_positions(initial_parameters)

        self.default_ee_rotation = self.get_link_rotation()

        # Identify and ignore remaining collisions
        self.ignore_collisions()

        # Final collision check
        for obj1, obj2 in self.wc.collisionTests():
            if obj1[1].collides(obj2[1]):
                print(f" - Object {obj1[0].getName()} collides with {obj2[0].getName()}")

        if self.robot.selfCollides():
            print("[init_robot] Robot self-collision detected")


    def ignore_collisions(self, debug=False, robot_link_test=False):
        ignored_collisions = set()
        for obj1, obj2 in self.wc.collisionTests():
            if obj1[1].collides(obj2[1]):
                ignored_collisions.add((obj1[0], obj2[0]))
                if debug:
                    print(f"[init_robot] Object {obj1[0].getName()} collides with {obj2[0].getName()}")

        for obj1, obj2 in ignored_collisions:
            self.wc.ignoreCollision((obj1, obj2))

            if robot_link_test:
                is_robot_link =  isinstance(obj1, klampt.robotsim.RobotModelLink) and isinstance(obj2, klampt.robotsim.RobotModelLink)
            else: is_robot_link = True

            if is_robot_link:
                self.robot.enableSelfCollision(obj1.getIndex(), obj2.getIndex(), False)


    def update_robot_cspace_and_colliders(self):
        """
        Updates the robot's configuration space and collision settings.
        """
        self.robot = self.world.robot("ur5")
        self.tool_tip = self.robot.link("tool_tip")
        self.wc = collide.WorldCollider(self.world)
        self.space = RobotCSpace(self.robot, self.wc)

        # Exclude collisions between fingertips and point cloud
        exclusion_links = [
            "left_inner_finger", "left_inner_finger_pad",
            "right_inner_finger", "right_inner_finger_pad", "dummy_camera_link",
            "base_link", "shoulder_link", "upper_arm_link", "forearm_link"
        ]

        exclusion_links = ["left_inner_finger_pad", "right_inner_finger_pad", "dummy_camera_link",
            "base_link", "shoulder_link", "upper_arm_link", "forearm_link"
        ]


        exclusion_terrains = [
            "ground_plane", "table", "shelf_base",
            "bottom_shelf", "top_shelf", "top_wall", "right_wall",
            "left_wall", "back_wall"
        ]

        exclusion_shelves = ["right_wall", "left_wall", "back_wall", "mid_low_shelf"]

        for i in range(self.world.numRigidObjects()):
            obj = self.world.rigidObject(i)
            for linkname in exclusion_links:
                link = self.world.robotLink("ur5", linkname)
                self.wc.ignoreCollision((obj, link))

        # Exclude collisions between specific terrains and robot links
        for terrainname in exclusion_terrains:
            terrain = self.world.terrain(terrainname)
            lim = -4 if terrainname in exclusion_terrains[-3:] else 0
            for linkname in exclusion_links[lim:]:
                link = self.world.robotLink("ur5", linkname)
                self.wc.ignoreCollision((terrain, link))

        # Exclude collisions between a specific rigid body and shelves
        if self.tm_rb_name is not None:
            for rb_name in exclusion_shelves:
                tm = self.world.rigidObject(self.tm_rb_name)
                rb = self.world.terrain(rb_name)
                self.wc.ignoreCollision((rb, tm))


    def add_heightmap_to_klampt(self, voxels, resolution=0.005, verbose=False):
        """
        Converts a 3D numpy array of voxels into a Geometry3D object.
        :param voxels: 3D numpy array where 1 indicates an occupied voxel.
        :param resolution: The size of each voxel.
        :return: Geometry3D object representing the voxel grid.
        """
        # transform voxel to mesh
        fill = (voxels > 0.7).astype(int)
        vertices, triangles = mcubes.marching_cubes(fill, 0.5)
        # simplify mesh
        mesh_simplifier = pyfqmr.Simplify()
        mesh_simplifier.setMesh(vertices, triangles)
        mesh_simplifier.simplify_mesh(target_count=len(triangles) / 30, aggressiveness=7, preserve_border=True,
                                      verbose=False)
        vertices, triangles, _ = mesh_simplifier.getMesh()
        invalid1 = triangles[:, 0] == triangles[:, 1]
        invalid2 = triangles[:, 0] == triangles[:, 2]
        invalid3 = triangles[:, 1] == triangles[:, 2]
        invalid = np.logical_or(invalid1, invalid2)
        invalid = np.logical_or(invalid, invalid3)
        triangles = triangles[np.logical_not(invalid)]
        # Extract vertices and triangles from the decimated mesh and set them to klamp'ts TriangleMesh object
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        tri_mesh.update_faces(tri_mesh.unique_faces())
        mesh = TriangleMesh()
        mesh.setVertices(tri_mesh.vertices.astype(np.int32))
        mesh.setIndices(tri_mesh.faces.astype(np.int32))
        geom = Geometry3D()
        geom.setTriangleMesh(mesh)
        geom.scale(resolution)
        geom.rotate(so3.rotation((0, 0, 1), np.deg2rad(90)))
        geom.translate([0.5, 0.52, 0.9])
        geom.setCollisionMargin(0.01)
        self.tm_rb = self.world.makeRigidObject('tm')
        self.tm_rb_name = 'tm'
        self.tm_rb.geometry().set(geom)
        self.tm_rb.appearance().setColor(1., 0.0, 0.0)
        if self.show_vis:
            vis.add("TriangleMesh", self.tm_rb)

        self.update_robot_cspace_and_colliders()

        self.init_robot()
        if verbose:
            print("[add_heightmap_to_klampt] Add Voxelmap to klampt")


    def delete_heightmap_from_klampt(self):
        self.world.remove(self.world.rigidObject(self.tm_rb_name))
        if (self.show_vis):
            vis.remove("TriangleMesh")
            vis.update()
        self.tm_rb = None
        self.tm_rb_name = None
        self.update_robot_cspace_and_colliders()
        self.init_robot()


    def check_config_feasibility(self, config, verbose=False):
        current_feasible = self.space.feasible(config)
        if not current_feasible:
            if verbose:
                print("[config_feasibility] given config not feasible!!")
                print(f"\n[config_feasibility] Config: {config}")
                print(
                    "[config_feasibility] The config feasibility is {}".format(
                        self.space.feasible(config)
                    )
                )
                env_collision = self.space.envCollision(config)
                print(f"[config_feasibility] The config Collision is {env_collision}")
                if env_collision:
                    failed_start = [
                        name
                        for (test, name) in zip(
                            self.space.feasibilityTests, self.space.feasibilityTestNames
                        )
                        if not (test(config))
                    ]
                    self.print_env_collision_check_info()
                    print("Start configuration failed tests:", ",".join(failed_start))
                print(
                    "[config_feasibility] The robot self collides {}".format(
                        self.robot.selfCollides()
                    )
                )
                print(
                    "[config_feasibility] The Robot is currently within its joint limits {} \n".format(
                        self.space.inJointLimits(config)
                    )
                )
                for obj1, obj2 in self.wc.collisionTests():
                    if obj1[1].collides(obj2[1]):
                        print(f" - Object {obj1[0].getName()} collides with {obj2[0].getName()}")

            return False
        if verbose:
            print("[config_feasibility] given config is feasible!!")
        return True


    def print_env_collision_check_info(self):
        for i, j in self.wc.collisionTests():
            if i[1].collides(j[1]):
                print(" - Object", i[0].getName(), "collides with", j[0].getName())


    def run_planner_default(
            self,
            plan: cspace.MotionPlan,
            maxIters: int = MAX_PLANNER_ITERS,
            maxTime: float = MAX_PLANNER_TIME,
            endpoints: Tuple = None,
            verbose: int = 1,
    ) -> Union[None, List[Config]]:
        """A default runner for a planner, works generally in a sane manner for
        most defaults and allows debugging by setting verbose >= 1.

        Args:
            plan (MotionPlan): a MotionPlan object at least partially configured.
            maxIters (int): the maximum number of iterations to run.
            maxTime (float): the maximum number of seconds to run.
            endpoints (None or pair): the endpoints of the plan, either Configs
                or goal specifications. If None uses the endpoints configured in
                plan.
            verbose (int): whether to print information about the planning.

        Returns:
            path or None: if successful,returns a path solving the terminal
            conditions specified in the plan.
        """
        if endpoints is not None:
            if len(endpoints) != 2:
                raise ValueError("Need exactly two endpoints")
            try:
                plan.setEndpoints(*endpoints)
            except RuntimeError:
                # must be invalid configuration
                if verbose:
                    if isinstance(endpoints[0], (list, tuple)) and isinstance(
                            endpoints[0][0], (float, int)
                    ):
                        print(
                            "Start configuration fails:",
                            plan.space.feasibilityFailures(endpoints[0]),
                        )
                    if isinstance(endpoints[1], (list, tuple)) and isinstance(
                            endpoints[1][0], (float, int)
                    ):
                        print(
                            "Goal configuration fails:",
                            plan.space.feasibilityFailures(endpoints[1]),
                        )
                return None
        # this is helpful to slightly speed up collision queries
        plan.space.cspace.enableAdaptiveQueries(True)
        plan.space.cspace.optimizeQueryOrder()

        t0 = time.time()

        # begin planning
        numIters = 0
        for round in range(maxIters // 10):
            plan.planMore(10)
            numIters += 10

            if not IS_PLANNER_OPTIMIZING:  # break on first path found
                path = plan.getPath()
                if path is not None and len(path) > 0:
                    break
            if time.time() - t0 > maxTime:
                break
        if verbose >= 1:
            print(
                "Planning time {}s over {} iterations".format(
                    time.time() - t0, numIters
                )
            )

        # this code just gives some debugging information. it may get expensive
        if verbose >= 2:
            V, E = plan.getRoadmap()
            print(len(V), "feasible milestones sampled,", len(E), "edges connected")

        if verbose >= 2:
            print("Planner stats:")
            print(plan.getStats())

        path = plan.getPath()
        if path is None or len(path) == 0:
            if verbose >= 1:
                print("Failed to plan feasible path")
                if verbose < 2:
                    print("Planner stats:")
                    print(plan.getStats())
                # debug some sampled configurations
                if verbose >= 2:
                    print("Some sampled configurations:")
                    print(V[0: min(10, len(V))])
            return None

        return path


    def find_ik(self, link, goal_pos, goal_rot, num_tries=5, verbose=False):
        start_config = self.robot.getConfig()
        solver = IKSolver(self.robot)

        obj = ik.objective(link, ref=None, R=goal_rot, t=goal_pos)
        solver.add(obj)

        # try to solve IK multiple times to get a good solution
        # vary the tolerance each time
        for i in range(num_tries):
            solver.setTolerance(1e-4 * 10 * i)
            res = solver.solve()
            if res:
                if verbose:
                    print(
                        f"[find_ik] IK succeeded after {i + 1} tries with tolerance {solver.getTolerance()}"
                    )
                break

        if not res:
            print("[find_ik] IK failed")
            print(f"[find_ik] ik residual: {solver.getResidual()}")

        goal_config = self.robot.getConfig()
        # print tcp position
        if verbose:
            print(
                f'[find_ik] tcp position: {self.robot.link("tool_tip").getTransform()[1]}'
            )

        self.robot.setConfig(start_config)

        return goal_config


    def get_link_transform(self, link):
        return self.robot.link(link).getTransform()


    def plan_to_cartesian_goal(
            self,
            link_name,
            goal_pos=None,
            goal_quat=None,
            settings=PLANNER_SETTINGS,
            num_ik_tries=5,
            verbose=False,
    ):
        """
        Args:
            link_name (str): name of the link to plan for
            goal_pos (list): goal position [x, y, z]
            goal_quat (list): goal orientation [x, y, z, w]
            settings (dict): planner settings
            verbose (bool): verbosity
        """
        # pdb.set_trace()
        if verbose:
            print("\n[plan_to_cartesian_goal] planning..")

        if goal_pos is None:
            raise ValueError("[plan_to_cartesian_goal] goal_pos must be provided")
        link = self.robot.link(link_name)

        # current pos
        current_pos = link.getTransform()[1]
        current_ori = link.getTransform()[0]
        current_ori = so3.quaternion(current_ori)

        if goal_quat is None:
            goal_rot = link.getTransform()[0]
        else:
            # convert quaternion to rotation matrix
            goal_rot = so3.from_quaternion(
                (goal_quat[3], goal_quat[0], goal_quat[1], goal_quat[2])
            )

        goal_config = self.find_ik(
            link, goal_pos, goal_rot, num_tries=num_ik_tries, verbose=verbose
        )
        init_config = self.robot.getConfig()
        init_config[10] = 0.79
        self.robot.setConfig(init_config)

        current_feasible = self.space.feasible(self.robot.getConfig())
        if not current_feasible:
            print(
                "\nThe current feasibility is {}".format(
                    self.space.feasible(self.robot.getConfig())
                )
            )
            print(
                "The current Collision is {}".format(
                    self.space.envCollision(self.robot.getConfig())
                )
            )
            print("The robot self collides {}".format(self.robot.selfCollides()))
            print(
                "The Robot is currently within its joint limits {} \n".format(
                    self.space.inJointLimits(self.robot.getConfig())
                )
            )

        goal_config[10] = 0.79
        goal_feasible = self.space.feasible(goal_config)
        if not goal_feasible:
            print(
                "\nThe goal config feasibility is {}".format(
                    self.space.feasible(goal_config)
                )
            )
            print(
                "This config Collision is {}".format(
                    self.space.envCollision(goal_config)
                )
            )
            print("The robot self collides {}".format(self.robot.selfCollides()))
            print(
                "The Robot is currently within its joint limits {} \n".format(
                    self.space.inJointLimits(goal_config)
                )
            )

        self.robot.setConfig(init_config)

        if verbose:
            print(
                f"[plan_to_cartesian_goal] current pose: {current_pos}, {current_ori}"
            )
            print(
                f"[plan_to_cartesian_goal] goal pose: {goal_pos}, {so3.quaternion(goal_rot)}"
            )
            print(f"[plan_to_cartesian_goal] start config: {self.robot.getConfig()}")
            print(f"[plan_to_cartesian_goal] goal config: {goal_config}")

        plan = plan_to_config(
            self.world, self.robot, goal_config, movingSubset="auto", **settings
        )

        if plan is None:
            print("[plan_to_cartesian_goal] Failed to get motion plan")
            return None

        numIters = 0
        for round in range(100):
            try:
                plan.planMore(50)
            except Exception as e:
                print(f"[plan_to_cartesian_goal] Exception: {e}")
                continue
            numIters += 1
            if plan.getPath() is not None:
                break

        path = plan.getPath()

        if path is None:
            print("[plan_to_cartesian_goal] Failed to plan path")
            return None

        #densify path
        traj = RobotTrajectory(self.robot, range(len(path)), path)
        #path = [traj.eval(t)
        #        for t in np.linspace(0, traj.getDuration(), len(path) * 300)]
        #print("HHHHAAAAAALLLLLOOOOO")
        if verbose:
            print(
                f"[plan_to_cartesian_goal] Motion planning successful with {len(path)} waypoints"
            )
            print("waypoints: ")
            for i in range(len(path)):
                print(f"  {i}: {path[i][:8]}")

        if (self.show_vis):
            vis.add("trajectory", traj)
        return path


    def plan_to_joint_goal(self, goal_config, check_feasibility=True, verbose=False, just_endpoints=False):
        current_config = self.robot.getConfig()
        start_config = copy.deepcopy(current_config)
        curr_conf = copy.deepcopy(current_config)
        goal_conf = copy.deepcopy(current_config)
        goal_conf[1:7] = goal_config[1:7]
        t0 = time.time()

        if check_feasibility:

            feasible_current = self.check_config_feasibility(current_config, verbose=verbose)
            feasible_goal = self.check_config_feasibility(goal_conf, verbose=verbose)

            if ((not feasible_goal) or (not feasible_current)):
                self.robot.setConfig(start_config)
                return None

        if(just_endpoints):
            return [start_config,goal_conf]
        if(just_endpoints):
            return [start_config,goal_conf]
        if(verbose):
            print(f"[plan_to_joint_goal] Time to check feasibility: {time.time() - t0}")
        t0 = time.time()
        plan = cspace.MotionPlan(self.space, **PLANNER_SETTINGS)
        plan.space.cspace.enableAdaptiveQueries(True)
        plan.space.cspace.optimizeQueryOrder()
        if (verbose):
            print(f"[plan_to_joint_goal] Time to setup planner: {time.time() - t0}")

        if verbose:
            print(f"[plan_to_joint_goal] current config: {curr_conf[:9]}")
            print(f"[plan_to_joint_goal] goal config: {goal_conf[:9]}")

        try:
            t0 = time.time()
            plan.setEndpoints(curr_conf, goal_conf)
            if (verbose):
                print(f"[plan_to_joint_goal] Time to set endpoints: {time.time() - t0}")
        except RuntimeError:
            # one of the configurations must be infeasible
            failed_start = [
                name
                for (test, name) in zip(
                    plan.space.feasibilityTests, plan.space.feasibilityTestNames
                )
                if not (test(start_config))
            ]
            for i, j in self.wc.collisionTests():
                if i[1].collides(j[1]) and verbose:
                    print(" - Object", i[0].getName(), "collides with", j[0].getName())
            if verbose:
                print("Start configuration failed tests:", ",".join(failed_start))
            failed_goal = [
                name
                for (test, name) in zip(
                    plan.space.feasibilityTests, plan.space.feasibilityTestNames
                )
                if not (test(goal_conf))
            ]
            for i, j in self.wc.collisionTests():
                if i[1].collides(j[1]) and verbose:
                    print(" - Object", i[0].getName(), "collides with", j[0].getName())
            if verbose:
                print("Goal configuration failed tests:", ",".join(failed_goal))
            # self.check_config_feasibility(goal_conf, verbose=True)
            self.robot.setConfig(start_config)
            return None

        path = None

        numIters = 0
        for round in range(100):
            try:
                plan.planMore(50)
            except Exception as e:
                print(f"[plan_to_joint_goal] Exception: {e}")
                continue
            numIters += 1
            path = plan.getPath()
            if path is not None:
                break

        if path is None:
            print("[plan_to_joint_goal] Failed to plan path")
            self.robot.setConfig(start_config)
            return None

        if verbose:
            print(
                f"[plan_to_joint_goal] Motion planning successful with {len(path)} waypoints"
            )
            print("    waypoints: ")
            for i in range(len(path)):
                print(f"       {i}: {path[i][:8]}")
        t0 = time.time()
        traj = RobotTrajectory(self.robot, range(len(path)), path)
        if (verbose):
            print(f"[plan_to_joint_goal] Time to create trajectory: {time.time() - t0}")
        if (self.show_vis):
            vis.add("trajectory", traj)

        self.robot.setConfig(start_config)
        return path


    def discretize_trajectory(self, trajectory, dt):
        tj = RobotTrajectory(self.robot, times=np.arange(len(trajectory)), milestones=trajectory)
        discrete = tj.discretize(dt=dt)
        return discrete.milestones


    def get_fk_of_links(self, joint_positions, use_link_indices=None):
        initial_config = self.robot.getConfig()

        self.robot.setConfig(joint_positions)

        link_positions = []
        if use_link_indices is None:
            use_link_indices = self.arm_indices + self.gripper_indices

        for link in use_link_indices:
            link_positions.append(self.robot.link(link).getTransform()[1])

        self.robot.setConfig(initial_config)

        return link_positions


    def get_fk_of_links_alternative(self, trajectory, link_names=None):
        if link_names is None:
            link_names = self.useful_swept_volume_names

        tj = RobotTrajectory(self.robot, np.arange(len(trajectory)), trajectory)

        all_link_poses = []
        for link_name in link_names:
            link_poses = np.array(tj.getLinkTrajectory(link_name).milestones)[:, -3:]
            all_link_poses.append(link_poses)
        all_link_poses = np.array(all_link_poses).transpose()
        all_link_poses = np.moveaxis(all_link_poses, [0, 1, 2], [2, 0, 1])
        return all_link_poses


    def set_joint_positions(self, joint_positions):
        if len(joint_positions) < 6:
            raise ValueError(
                "The joint_positions list must contain at least 6 elements."
            )
        current_config = self.robot.getConfig()
        current_config[1:7] = joint_positions[:6]

        # set the gripper joints based on the index mapping
        for pb_index, klampt_index in self.pb_to_klampt_gripper_joint_map.items():
            current_config[klampt_index] = joint_positions[pb_index]

        self.robot.setConfig(current_config)

    def test_feasibility(self, start_config=None, target_pose=None, target_rotation=None,
                         target_direction=None, just_endpoints=False, target_link=None,
                         is_pybullet_config=False, verbose=False, free_yaw=False):

        if target_pose is None:
            print('Pushing Utils: target pose is None!?')
            return None, None, None

        if target_link is None:
            target_link = self.robot.link('tool_tip')

        if target_rotation is None:
            target_rotation = self.default_ee_rotation

        if start_config is None:
            start_config = self.default_initial_parameters

        if is_pybullet_config:
            self.set_joint_positions(start_config)
            start_config = copy.deepcopy(self.robot.getConfig())
        else:
            self.robot.setConfig(start_config)

        # Build IK objective
        if target_direction is not None:
            # View direction constraint
            camera_origin = np.array(target_pose[:3])
            camera_target = np.array(target_direction[:3])
            direction = camera_target - camera_origin
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0:
                raise ValueError("Camera origin and target are the same. Cannot define a view direction.")
            direction_unit = direction / direction_norm

            local = [[0, 0, 0], [0, 0, direction_norm]]
            world = [camera_origin.tolist(), (camera_origin + direction).tolist()]
            goal = ik.objective(target_link, local=local, world=world)




        elif free_yaw:
            R_mat = np.array(target_rotation).reshape(3, 3)

            pos = np.array(target_pose.get()[:3])

            # Constraint: position and Z axis direction only

            local = [[0, 0, 0], [0, 0, 1]]

            world = [pos.tolist(), (pos + R_mat[:, 2]).tolist()]

            goal = ik.objective(target_link, local=local, world=world)


        else:
            # Full pose constraint
            goal = ik.objective(target_link, R=target_rotation, t=target_pose[:3])

        success = ik.solve(goal, activeDofs=[1, 2, 3, 4, 5, 6])
        if not success:
            return None, None, None

        target_joint_config = self.robot.getConfig()
        target_link_rotation, target_link_position = target_link.getTransform()
        r = R.from_matrix(np.array(target_link_rotation).reshape(3, 3))
        target_link_euler_angles = r.as_euler('xyz', degrees=False)

        self.robot.setConfig(start_config)
        path = self.plan_to_joint_goal(target_joint_config, verbose=verbose, just_endpoints=just_endpoints)
        if path is not None:
            return target_joint_config, path, [target_link_position, target_link_euler_angles]
        return None, None, None

    def get_link_rotation(self):
        # Get rotation matrix of a specific link (e.g., end-effector)
        link_index = 21  # tool-tip
        link = self.robot.link(link_index)
        R, t = link.getTransform()  # R is the 3x3 rotation matrix, t is the translation vector
        return R



    def visualize_path(self, path):
        traj = RobotTrajectory(self.robot, range(len(path)), path)
        # resource.edit("Planned trajectory", traj, world=self.world)

        # Here's another way to do it: visualize path in the vis module
        from klampt import vis
        if (self.show_vis):
            vis.add("world", self.world)
            vis.animate(("world", self.robot.getName()), path)
            vis.add("trajectory", traj)
            vis.spin(float("inf"))


    def visualize(self):
        if (self.show_vis):
            vis.add("world", self.world)
            link = self.robot.link("dummy_camera_link")
            vis.add("cam", link.getTransform() )
            vis.show()
            vis.run()


    def print_robot_info(self):
        print("Robot info:")
        print("  Name:", self.robot.getName())
        print("  Links:", self.robot.numLinks())
        print("  Link names:")
        for i in range(self.robot.numLinks()):
            print(f"    {i}: {self.robot.link(i).getName()}")
        print("  Joints:", self.robot.numDrivers())
        print("  Joint names:")
        for i in range(self.robot.numDrivers()):
            print("    ", self.robot.driver(i).getName())


if __name__ == "__main__":
    klampt_utils = KlamptUtils()
    klampt_utils.print_robot_info()
    klampt_utils.show_vis = True
    klampt_utils.visualize()
"""
Code is based on
https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs
and
https://github.com/openai/gym/tree/master/gym/envs
"""

import pybullet as p
from pybullet_utils import bullet_client
import gymnasium as gym
from gymnasium.utils import EzPickle
import pybullet_data
import numpy as np 
import pkgutil
import sys

class BasePybulletEnv(gym.Env):
    def __init__(self, render=False, shared_memory=False, hz=240):
        EzPickle.__init__(**locals())
        """Initialize the BasePybullet environment.

                Args:
                - render: Whether to render the environment.
                - shared_memory: If the simulation should use shared memory.
                - hz: Frequency of the simulation.
                - use_egl: If the environment should use EGL for rendering.
                """
        self.step_size_fraction = hz
        self.shared_memory = shared_memory
        self.render = render
        self._p = None
        self.pybullet_init()
        self._set_action_space()
        self._set_observation_space()

    def _reset_base_simulation(self):
        """Reset the basic simulation settings."""
        self._p.resetSimulation()
        self._p.setGravity(0, 0, -9.81)

    def pybullet_init(self):
        render_option = p.DIRECT
        if self.render:
            print("I will render")
            render_option = p.GUI

        self._p = bullet_client.BulletClient(connection_mode=render_option)
        self._urdfRoot = pybullet_data.getDataPath()
        self._p.setAdditionalSearchPath(self._urdfRoot)

        self._egl_plugin = None
        if not self.render:
            print("I will use the alternative renderer")
            assert sys.platform == 'linux', ('EGL rendering is only supported on ''Linux.')
            egl = pkgutil.get_loader('eglRenderer')
            if egl:
                self._egl_plugin = self._p.loadPlugin(egl.get_filename(), '_eglRendererPlugin')
            else:
                self._egl_plugin = self._p.loadPlugin('eglRendererPlugin')
            print('EGL renderering enabled.')

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setPhysicsEngineParameter(enableFileCaching=0)
        self._p.setPhysicsEngineParameter(numSolverIterations=200)
        self._p.setTimeStep(1. / self.step_size_fraction)

        if self.render:
            self._p.resetDebugVisualizerCamera(
                cameraDistance=1.,
                cameraYaw=17.999942779541016,
                cameraPitch=-43.99998092651367,
                cameraTargetPosition=[0.05872843414545059, 0.4925108850002289, 0.9959999918937683]
            )

        self.client_id = self._p._client
        self._reset_base_simulation()

        control_frequency = 15
        self.per_step_iterations = int(self.step_size_fraction / control_frequency)
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)


    def step_simulation(self, num_steps):
        """Perform a simulation step.

                Args:
                - num_steps: Number of steps to simulate.
                """
        for _ in range(int(num_steps)):
            self._p.stepSimulation(physicsClientId=self.client_id)
            self._p.performCollisionDetection()


    def collision_checks(self, collision, closest_check, body_A, body_B, link_A=-1, link_B=-1):
        collision += self._p.getContactPoints(bodyA=body_A, bodyB=body_B, linkIndexA=link_A, linkIndexB=link_B,
                                              physicsClientId=self.client_id)
        closest_check += self._p.getClosestPoints(bodyA=body_A, bodyB=body_B, linkIndexA=link_A, linkIndexB=link_B,
                                                  distance=0.001, physicsClientId=self.client_id)
        return collision, closest_check


    def close(self):
        """Close the environment and cleanup."""
        if self._egl_plugin is not None:
            p.unloadPlugin(self._egl_plugin)
        self._p.disconnect()


    def step(self):
        """Step function to be implemented by subclasses."""
        pass


    def reset(self):
        """Reset function to be implemented by subclasses."""
        pass


    def _set_action_space(self):
        """
        Define the dummy action space for the environment.
        """
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)# x, y, z, angle


    def _set_observation_space(self):
        """
        Define the dummy observation space for the environment.
        """
        self.observation_space = gym.spaces.Box(-np.ones(1), np.ones(1), dtype=np.float32)

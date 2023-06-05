"""This file implements jumping on a quadruped with reference from trajectory optimization.
"""

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import time, datetime
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet_data
import pandas as pd
from pkg_resources import parse_version
import re

# Should only be using A1 for jumping

import usc_learning.envs.quadruped_master.configs_a1 as robot_config
from usc_learning.envs.quadruped_master import quadruped_gym_env  # import QuadrupedGymEnv
import usc_learning.envs.quadruped_master.quadruped as quadruped
from usc_learning.imitation_tasks.traj_motion_data import TrajTaskMotionData
import usc_learning
from usc_learning.envs.quadruped_master.quadruped_gym_env import VIDEO_LOG_DIRECTORY

opt_data = 'data9_forward/jumpingFull_A1_1ms_h00_d60.csv'
opt_trajs_path = 'double_backflip'
# opt_trajs_path = 'backflip_d-60'

print('*' * 80)
# print('opt trajs path', opt_trajs_path)

TEST_TIME_HEIGHTS_F_R = np.array([[0.005, 0.005, 0.05, 0.005, 0.05, 0.04, 0.04, 0.04, 0.03],
                                  [0.005, 0.005, 0.005, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]]).T

TEST_TIME_HEIGHTS_F_R = np.array([[0.01, 0.01],
                                  [0.01, 0.01]]).T
TEST_TIME_HEIGHTS_F_R = np.array([[0.01, 0.05, 0.01, 0.1],
                                  [0.05, 0.01, 0.1, 0.01]]).T
TEST_IDX = 0
TEST_TIME_HEIGHTS_F_R = []


ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01

FLIGHT_TIME_START = 0.8

CONTROL_MODES = ["TORQUE", "PD", "IMPEDANCE"]

TASK_MODES = ["DEFAULT",
              "DEFAULT_CARTESIAN"]

TASK_ENVS = ["FULL_TRAJ"]

class Normalizer():
    """ this ensures that the policy puts equal weight upon
        each state component.
    """

    # Normalizes the states
    def __init__(self, state_dim):
        """ Initialize state space (all zero)
        """
        self.state = np.zeros(state_dim)
        self.mean = np.zeros(state_dim)
        self.mean_diff = np.zeros(state_dim)
        self.var = np.zeros(state_dim)

    def observe(self, x):
        """ Compute running average and variance
            clip variance >0 to avoid division by zero
        """
        self.state += 1.0
        last_mean = self.mean.copy()

        # running avg
        self.mean += (x - self.mean) / self.state

        # used to compute variance
        self.mean_diff += (x - last_mean) * (x - self.mean)
        # variance
        self.var = (self.mean_diff / self.state).clip(min=1e-2)

    def normalize(self, states):
        """ subtract mean state value from current state
            and divide by standard deviation (sqrt(var))
            to normalize
        """
        state_mean = self.mean
        state_std = np.sqrt(self.var)
        return (states - state_mean) / state_std


class ImitationGymEnv(quadruped_gym_env.QuadrupedGymEnv):
    """ The imitation gym environment for a quadruped. """
    NUM_SUBSTEPS = 10

    def __init__(self,
                 robot_config=robot_config,
                 traj_filename=opt_data,
                 use_multiple_trajs=True,  # if use multiple trajectories
                 opt_trajs_path=opt_trajs_path,
                 useTrajCartesianData=True,  # whether to use Cartesian PD control - allows better tracking
                 time_step=0.001,  # 0.01
                 set_kp_gains="HIGH",  # either LOW / HIGH
                 motor_control_mode="TORQUE",
                 # task_env= "FULL_TRAJ",
                 # task_mode="DEFAULT_CARTESIAN",
                 task_mode="DEFAULT",
                 land_and_settle=False,
                 # traj_num_lookahead_steps=4,
                 accurate_motor_model_enabled=True,
                 hard_reset=True,
                 render=True,
                 record_video=True,
                 env_randomizer=None,
                 randomize_dynamics=True,
                 # add_terrain_noise=False,
                 test_index=TEST_IDX,
                 test_heights=TEST_TIME_HEIGHTS_F_R,
                 test_mass=None,


                 **kwargs  # any extras from legacy
                 ):
        """
          Args:
          robot_config: The robot config file, should be A1 for jumping.
          traj_filename: Which traj to try to track.
          enable_clip_motor_commands: TODO\
          time_ste: Sim time step. WILL BE CHANGED BY NUM_SUBSTEPS.
          motor_control_mode: Whether to use torque control, PD, control, etc.
          task_mode: See descriptions above. Execute traj, add to traj, or pure RL.
          traj_num_lookahead_steps: how many future trajectory points to include
            (see get_observation for details on which)
          accurate_motor_model_enabled: Whether to use the accurate DC motor model.
            Should always be true.
          control_latency: TODO.
          hard_reset: Whether to wipe the simulation and load everything when reset
            is called. If set to false, reset just place the quadruped back to start
            position and set its pose to initial configuration.
          on_rack: Whether to place the quadruped on rack. This is only used to debug
            the walking gait. In this mode, the quadruped's base is hanged midair so
            that its walking gait is clearer to visualize.
          enable_action_interpolation: TODO
          enable_action_filter: TODO
          render: Whether to render the simulation.
          env_randomizer: TO ADD. An EnvRandomizer to randomize the physical properties
            during reset().
        """
        self._sensor_records = None
        self._robot_states = []
        self._test_mass = test_mass
        rand_front_box = 0.05,
        rand_rear_box = 0,

        def get_opt_trajs(path):
            """ Returns all optimal trajectories in directory. """
            # try:
            #   files =  os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), path)
            # except:
            path = os.path.join(os.path.dirname(os.path.abspath(usc_learning.imitation_tasks.__file__)), path)
            files = os.listdir(path)
            paths = [os.path.join(path, basename) for basename in files if
                     (basename.endswith('.csv') and 'cartesian' not in basename)]
            return paths

        self._use_multiple_trajs = use_multiple_trajs
        if use_multiple_trajs:
            self.opt_traj_filenames = get_opt_trajs(opt_trajs_path)
            self.num_opt_trajs = len(self.opt_traj_filenames)
            self.curr_opt_traj = 0
            self._traj_task = TrajTaskMotionData(
                filename=self.opt_traj_filenames[self.curr_opt_traj % self.num_opt_trajs], dt=0.001,
                useCartesianData=useTrajCartesianData)
            # iterate through each trajectory to get max/min observations
            traj_task_max_range = np.amax(self._traj_task.full_state, axis=1)
            traj_task_min_range = np.amin(self._traj_task.full_state, axis=1)
            for i in range(self.num_opt_trajs):
                traj_task = TrajTaskMotionData(filename=self.opt_traj_filenames[i], dt=0.001,
                                               useCartesianData=useTrajCartesianData)
                traj_task_max_range = np.maximum(traj_task_max_range, np.amax(traj_task.full_state, axis=1))
                traj_task_min_range = np.minimum(traj_task_min_range, np.amin(traj_task.full_state, axis=1))
            self.traj_task_min_range = traj_task_min_range
            self.traj_task_max_range = traj_task_max_range
        else:
            # desired trajectory to track (need to initialize first)
            self._traj_task = TrajTaskMotionData(filename=traj_filename, dt=0.001,
                                                 useCartesianData=useTrajCartesianData)
            self.traj_task_max_range = np.amax(self._traj_task.full_state, axis=1)
            self.traj_task_min_range = np.amin(self._traj_task.full_state, axis=1)
        self._useTrajCartesianData = useTrajCartesianData
        self._task_mode = task_mode
        # self._traj_num_lookahead_steps = traj_num_lookahead_steps
        self._traj_task_index = 0
        self._motor_control_mode = motor_control_mode
        # self._observation_space_mode = observation_space_mode
        # self._task_env = task_env
        self._set_kp_gains = set_kp_gains
        self._land_and_settle = land_and_settle
        self.TEST_TIME_HEIGHTS_F_R = test_heights
        self.TEST_IDX = test_index
        if land_and_settle:
            self.opt_traj_filenames = get_opt_trajs(
                'test_data2')  # get_opt_trajs('test_data') # test_data2 test_paper_vid
            self.num_opt_trajs = len(self.opt_traj_filenames)
            self.curr_opt_traj = 0

        self.save_real_base_xyz = []
        self.save_traj_base_xyz = []
        self.save_real_base_rpy = []
        self.save_traj_base_rpy = []

        super(ImitationGymEnv, self).__init__(
            urdf_root=pybullet_data.getDataPath(),  # pyb_data.getDataPath() # careful_here
            robot_config=robot_config,
            time_step=time_step,
            # task_env=task_env,
            accurate_motor_model_enabled=accurate_motor_model_enabled,
            motor_control_mode=motor_control_mode,
            hard_reset=hard_reset,
            render=render,
            record_video=record_video,
            env_randomizer=env_randomizer,
            randomize_dynamics=randomize_dynamics)
    ######################################################################################
    # Reset
    ######################################################################################
    def reset(self):
        if self._hard_reset:
            # print('hard reset')
            self._pybullet_client.resetSimulation()
            self._pybullet_client.setPhysicsEngineParameter(
                numSolverIterations=int(self._num_bullet_solver_iterations))
            self._pybullet_client.setTimeStep(self._time_step)
            self.plane = plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
            self._pybullet_client.changeVisualShape(plane, -1, rgbaColor=[1, 1, 1, 0.9])

            # self.add_box()
            # self.add_box2()

            if self._use_multiple_trajs:
                # load next traj
                # self.curr_opt_traj += 1
                # if self.curr_opt_traj >= self.num_opt_trajs:
                #   self.curr_opt_traj = 0
                self.curr_opt_traj = np.random.randint(0, self.num_opt_trajs)
                # print("curr_opt_traj:", self.curr_opt_traj)
                self._traj_task = TrajTaskMotionData(filename=self.opt_traj_filenames[self.curr_opt_traj], dt=0.001,
                                                     useCartesianData=self._useTrajCartesianData)
                # add box according to height, distance
                # _CHASSIS_NAME_PATTERN = re.compile(r"\w*floating_base\w*")
                curr_traj_name = os.path.basename(self.opt_traj_filenames[self.curr_opt_traj])
                # jump_height = float(curr_traj_name[20:22]) / 100
                # jump_dist = float(curr_traj_name[24:26]) / 100
                # print('Current traj is:', curr_traj_name)

                # means we are in 3 digit, 100
                # if jump_dist < .3:
                #     jump_dist = float(curr_traj_name[24:27]) / 100
                # if self._is_render and jump_dist > 0.5:
                #     print('Current traj is:', curr_traj_name)
                #     print('Adding box at h', jump_height, 'd', jump_dist)
                #     self.add_box_at(jump_height, jump_dist)
                self.add_box_at(1, 0)

            else:
                if self._is_render:
                    self.add_box_at(0.01, 0.6)

            self._robot_config.INIT_POSITION = [0, 0, 0.2+1]
            rand_front = 0
            rand_rear = 0
            if self._randomize_dynamics:
                    mult = 0.0
                    rand_front = mult * np.random.random()
                    rand_rear = mult * np.random.random()
                    if self._land_and_settle:
                        self.TEST_IDX = self.TEST_IDX % len(self.TEST_TIME_HEIGHTS_F_R)
                        rand_front = self.TEST_TIME_HEIGHTS_F_R[self.TEST_IDX, 0]
                        rand_rear = self.TEST_TIME_HEIGHTS_F_R[self.TEST_IDX, 1]
                        self.TEST_IDX += 1
                        print('======== USING HEIGHTS', rand_front, rand_rear, "TEST IDX", self.TEST_IDX)
                    self.add_box_ff(z_height=rand_front)
                    self.add_box_rr(z_height=rand_rear)

                    min_mu = 0.6
                    ground_mu_k_front = min_mu + (1 - min_mu) * np.random.random()*0
                    ground_mu_k_rear = min_mu + (1 - min_mu) * np.random.random()*0
                    self._pybullet_client.changeDynamics(self.box_ff, -1, lateralFriction=ground_mu_k_front)
                    self._pybullet_client.changeDynamics(self.box_rr, -1, lateralFriction=ground_mu_k_rear)

            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
            self._pybullet_client.setGravity(0, 0, -10)
            acc_motor = self._accurate_motor_model_enabled
            motor_protect = self._motor_overheat_protection
            self._robot = (quadruped.Quadruped(pybullet_client=self._pybullet_client,
                                               robot_config=self._robot_config,
                                               time_step=self._time_step,
                                               self_collision_enabled=self._self_collision_enabled,
                                               motor_velocity_limit=self._motor_velocity_limit,
                                               pd_control_enabled=self._pd_control_enabled,
                                               accurate_motor_model_enabled=acc_motor,
                                               motor_control_mode=self._motor_control_mode,
                                               motor_overheat_protection=motor_protect,
                                               on_rack=self._on_rack,
                                               render=self._is_render))


        else:
            # print('soft reset')
            self._robot.Reset(reload_urdf=False)
            # self._load_plane()

        if self._env_randomizer is not None:
            raise ValueError('Not implemented randomizer yet.')
            self._env_randomizer.randomize_env(self)

        if self._randomize_dynamics and self._test_mass is None:
            if np.random.random() < 0.6:  # 0.1:#0.6: # don't always put in a mass..
                # self._robot.RandomizePhysicalParams(mass_percent_change=0.1,base_only=False) # base_only=True
                self._robot.RandomizePhysicalParams(mass_percent_change=0.2, base_only=False)  # base_only=True
            # if np.random.random() < 0.8:
            #     # pass
            #     print(".")
            #     self._add_base_mass_offset()
        elif self._test_mass is not None:
            self._add_base_mass_offset(spec_mass=self._test_mass[0], spec_location=self._test_mass[1:])

        base_pos = self._robot.GetBasePosition()
        # print('base pos before anything happens', self._robot.GetBasePosition())
        base_pos = list(base_pos)

        base_orn = self._robot.GetBaseOrientation()
        self._robot.ResetBasePositionAndOrientation(base_pos, base_orn)
        if self._is_render:
            time.sleep(1)

        # print('base pos after reset to max + init', self._robot.GetBasePosition())
        # set initial condition from traj
        des_state = self._traj_task.get_state_at_index(0)
        des_torques = self._traj_task.get_torques_at_index(0)
        q_des = self._traj_task.get_joint_pos_from_state(des_state)
        self._robot.SetJointStates(np.concatenate((q_des, [0] * 12)))

        ground_mu_k = 0.6  # 0.5#+0.5*np.random.random()
        self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=ground_mu_k)

        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._traj_task_index = 0
        self._objectives = []
        # self.save_real_base_xyz = []
        # self.save_traj_base_xyz = []
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                         self._cam_pitch, [0, 0, 0])
        # set back..
        self._settle_robot()  # apply joint PD controller to joint

        # print('base pos after init', self._robot.GetBasePosition())

        if self._is_record_video:
            self.recordVideoHelper()

        # self._last_action = self._robot._applied_motor_torque
        self._last_base_euler = self._robot.GetBaseOrientationRollPitchYaw()

        if np.any(np.abs(self._last_base_euler) > .5):
            # means we probably fell over
            print('Bad start, reset. Pos is', self._robot.GetBasePosition(), 'ORN', self._last_base_euler)
            self.reset()

    def _settle_robot(self):
        """ Settle robot and add noise to init configuration. """
        # change to PD control mode to set initial position, then set back..
        # TODO: make this cleaner

        des_state = self._traj_task.get_state_at_index(0)
        des_torques = self._traj_task.get_torques_at_index(0)
        q_des = self._traj_task.get_joint_pos_from_state(des_state)
        self._robot.SetJointStates(np.concatenate((q_des, [0] * 12)))

        tmp_save_motor_control_mode_ENV = self._motor_control_mode
        tmp_save_motor_control_mode_ROB = self._robot._motor_control_mode
        tmp_save_motor_control_mode_MOT = self._robot._motor_model._motor_control_mode
        self._motor_control_mode = "PD"
        self._robot._motor_control_mode = "PD"
        self._robot._motor_model._motor_control_mode = "PD"
        if self._motor_control_mode != "TORQUE":
            for _ in range(600):
                if self._pd_control_enabled or self._accurate_motor_model_enabled:
                    # self._robot.ApplyAction([math.pi / 2] * 8)
                    # self._robot.ApplyAction(self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS)
                    self._robot.ApplyAction(q_des)
                    # print('Motor angles', self._robot.GetMotorAngles())
                    # print('Motor torques', self._robot.GetMotorTorques())
                self._pybullet_client.stepSimulation()
                if self._is_render:
                    time.sleep(0.01)
        # set control mode back
        self._motor_control_mode = tmp_save_motor_control_mode_ENV
        self._robot._motor_control_mode = tmp_save_motor_control_mode_ROB
        self._robot._motor_model._motor_control_mode = tmp_save_motor_control_mode_MOT

        # set some init noise
        def _noise(n, delta):
            return delta * np.random.random(n) - delta / 2

        self._robot.ApplyAction(q_des + _noise(12, 0))
        self._pybullet_client.stepSimulation()

    def _add_base_mass_offset(self, spec_mass=None, spec_location=None):
        quad_base = np.array(self._robot.GetBasePosition())
        quad_ID = self._robot.quadruped
        offset_low = np.array([-0.05, -0.02, 0])
        offset_upp = np.array([0.05, 0.02, 0])
        # offset_low = np.array([0, 0, 0])
        # offset_upp = np.array([0, 0, 0])
        #   block_pos_delta_base_frame = -1*np.array([-0.2, 0.1, -0.])
        if spec_location is None:
            block_pos_delta_base_frame = self.scale_rand(3, offset_low, offset_upp)
        else:
            block_pos_delta_base_frame = np.array(spec_location)
        if spec_mass is None:
            base_mass = 3 * np.random.random()
        else:
            base_mass = spec_mass
        if self._is_render:
            print('=========================== Random Mass:')
            print('Mass:', base_mass, 'location:', block_pos_delta_base_frame)

            # if rendering, also want to set the halfExtents accordingly
            # 1 kg water is 0.001 cubic meters
            boxSizeHalf = [(base_mass * 0.001) ** (1 / 3) / 2] * 3
            # boxSizeHalf = [0.05]*3
            translationalOffset = [0, 0, 0.1]
        else:
            boxSizeHalf = [0.05] * 3
            translationalOffset = [0] * 3

        # sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX, halfExtents=[0.05]*3)
        sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX, halfExtents=boxSizeHalf,
                                                               collisionFramePosition=translationalOffset)
        # orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
        base_block_ID = self._pybullet_client.createMultiBody(baseMass=base_mass,
                                                              baseCollisionShapeIndex=sh_colBox,
                                                              basePosition=quad_base + block_pos_delta_base_frame,
                                                              # basePosition = quad_base,
                                                              baseOrientation=[0, 0, 0, 1])

        cid = self._pybullet_client.createConstraint(quad_ID, -1, base_block_ID, -1, self._pybullet_client.JOINT_FIXED,
                                                     [0, 0, 0], [0, 0, 0], -block_pos_delta_base_frame)
        # disable self collision between box and each link
        for i in range(-1, self._pybullet_client.getNumJoints(quad_ID)):
            self._pybullet_client.setCollisionFilterPair(quad_ID, base_block_ID, i, -1, 0)

    ######################################################################################
    # Step
    ######################################################################################

    def _opt_traj_pd_controller(self, q_des, q, dq_des, dq, tau):
        """ Quick test controller to track opt traj, from sim_opt_traj_v2.py """
        kps = np.asarray([300, 300, 300] * 4)
        kds = np.asarray([3, 3, 3] * 4)

        if self._set_kp_gains == "HIGH":
            # this may be better
            kps = self._robot_config.kps
            kds = self._robot_config.kds
        elif self._set_kp_gains == "LOW":
            kps = np.asarray([100, 100, 100] * 4)
            kds = np.asarray([2, 2, 2] * 4)
        else:
            raise ValueError('check kp gains')

        return kps * (q_des - q) + kds * (dq_des - dq) - tau

    def _opt_traj_cartesian_pd_controller(self, foot_pos, foot_vel, foot_force):
        """ Compute Cartesian PD controller gains.

        FOR NOW:
        -ignore foot_force from opt
        -use default kpCartesian and kdCartesian as in normal impedance control, can update these as go on
        """
        actions = np.zeros(12)
        for j in range(4):
            actions[j * 3:j * 3 + 3] = self._robot._ComputeLegImpedanceControlActions(foot_pos[j * 3:j * 3 + 3],
                                                                                      foot_vel[j * 3:j * 3 + 3],
                                                                                      # np.diag([800,800,800]), #self._robot_config.kpCartesian
                                                                                      # np.diag([30,30,30]), # self._robot_config.kdCartesian
                                                                                      np.diag([500, 500, 500]),
                                                                                      # self._robot_config.kpCartesian
                                                                                      np.diag([10, 10, 10]),
                                                                                      # self._robot_config.kdCartesian
                                                                                      np.zeros(3),
                                                                                      j)
        return actions

    def _get_default_imitation_action(self):
        # should this be get_sim_time + some dt ahead to line up correctly?
        # TODO: verify time
        curr_time = self.get_sim_time() + self._time_step


            # desired trajectory state at this time
        traj_base_pos, traj_base_orn, traj_base_linvel, \
        traj_base_angvel, traj_joint_pos, traj_joint_vel = self.get_traj_state_at_time(curr_time)
            # feedforward torques
        des_torques = self._traj_task.get_torques_at_time(curr_time)

        if self._is_render:
            # current robot state
            curr_base_pos, curr_base_orn, curr_base_linvel, \
            curr_base_angvel, curr_joint_pos, curr_joint_vel = self.get_robot_state()

            self.save_traj_base_xyz.extend(traj_base_pos)
            self.save_real_base_xyz.extend(curr_base_pos)
            self.save_traj_base_rpy.extend(traj_base_orn)
            self.save_real_base_rpy.extend(curr_base_orn)

        # des_state = traj_task.get_state_at_index(i)
        # des_torques = self._traj_task.get_torques_at_index(i)
        q_des = traj_joint_pos
        dq_des = traj_joint_vel
        q = self._robot.GetMotorAngles()
        dq = self._robot.GetMotorVelocities()
        tau = des_torques
        return q_des, q, dq_des, dq, tau
        # return self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)

    def _get_cartesian_imitation_action(self):
        """ Get default foot pos/vel/force
        Returns: foot_pos, foot_vel, foot_force (each 12 values)
        """
        curr_time = self.get_sim_time() + self._time_step
        return self.get_cartesian_traj_state_at_time(curr_time)

    def _transform_action_to_motor_command(self):
        """ Get action depeding on mode.
        DEFAULT: just set sequentially with PD controller on des traj state
        IL:
        """
        action = np.zeros(12) # initialization
        if self._task_mode in TASK_MODES:
            # get optimal action from imitation data
            # note the opt action may be messed up from the additive torques we are supplying
            q_des, q, dq_des, dq, tau = self._get_default_imitation_action()
            # print("task_mode is:", self._task_mode)

            if self._task_mode == "DEFAULT":
                # quick test
                # q = self._robot.GetMotorAngles()
                # motor_commands_min = (q - self._time_step * self._robot_config.VELOCITY_LIMITS)
                # motor_commands_max = (q + self._time_step * self._robot_config.VELOCITY_LIMITS)
                # motor_commands = np.clip(q_des, motor_commands_min, motor_commands_max)
                # action = self._opt_traj_pd_controller(motor_commands, q, dq_des, dq, tau)
                action = self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)

            elif self._task_mode == "DEFAULT_CARTESIAN":
                # As in Cheetah 3 jumping paper
                # joint PD w torque
                tau_ff = self._opt_traj_pd_controller(q_des, q, dq_des, dq, tau)
                # Cartesian PD
                foot_pos, foot_vel, foot_force = self._get_cartesian_imitation_action()
                tau = self._opt_traj_cartesian_pd_controller(foot_pos, foot_vel, foot_force)
                action = tau + tau_ff
            else:
                raise NotImplementedError('Task mode not implemented.')

        return action

    def jump(self):

        # if self._is_render:
        #     self._render_step_helper()
        # self._dt_motor_torques = []
        # self._dt_motor_velocities = []
        # self._dt_torques_velocities = []
        self._base_poss = []
        self._base_orns = []
        self._base_poss_orns = []
        self._base_vel = []
        self._foot_forces = []
        self._foot_torque = []
        self._GRF_x = []
        self._GRF_y = []
        self._GRF_z = []
        perturbation = np.zeros(3)

        T = 1500 # total time of each jump (e.g 1200 ms)
        for i in range(T):
            self._robot.ApplyExternalForce(perturbation)
            proc_action = self._transform_action_to_motor_command()
            self._robot.ApplyAction_v2(proc_action)
            self._pybullet_client.stepSimulation()
            self._sim_step_counter += 1
            # self._dt_motor_torques.append(self._robot.GetMotorTorques())
            # self._dt_motor_velocities.append(self._robot.GetMotorVelocities())
            # self._base_poss.append(self._robot.GetBasePosition())
            # self._base_orns.append(self._robot.GetBaseOrientation())
            base_pos = np.concatenate((self._robot.GetBasePosition(), self._robot.GetBaseOrientationRollPitchYaw()))
            self._base_poss_orns.append(base_pos)
            # print("base_vl:", self._robot.GetBaseLinearVelocity())
            base_vl = np.concatenate(self._robot.GetBaseLinearAngularVelocity())
            print("base_vl:", base_vl)
            # print("base_vl:", self._robot.GetBaseLinearVelocity())
            self._base_vel.append(base_vl)

            _, _, GRF_x, GRF_y, GRF_z, feetInContactBool = self._robot.GetContactInfo()

            # force on each leg
            GRF_FR = np.array([GRF_x[0], GRF_y[0], GRF_z[0]])
            GRF_FL = np.array([GRF_x[1], GRF_y[1], GRF_z[1]])
            GRF_RR = np.array([GRF_x[2], GRF_y[2], GRF_z[2]])
            GRF_RL = np.array([GRF_x[3], GRF_y[3], GRF_z[3]])
            # print("GRF on front right:", GRF_FR)
            # print("GRF on front left:", GRF_FL)
            # print("GRF on rear right:", GRF_RR)
            # print("GRF on rear left:", GRF_RL)

            # print("Time:", i*0.001)
            # print("feetInContactBool:", feetInContactBool)
            # print("GRF_x:", GRF_x)
            # print("GRF_y:", GRF_y)
            # print("GRF_z:", GRF_z)

            f = np.concatenate((GRF_FR, GRF_FL, GRF_RR, GRF_RL))
            self._foot_forces.append(f)
            # self._foot_forces.append(f)
            # self._GRF_x.append(GRF_x)
            # self._GRF_y.append(GRF_y)
            # self._GRF_z.append(GRF_z)
            # f = np.concatenate((GRF_x, GRF_y, GRF_z))
            # self._foot_forces.append(f)

            # Leg order: FR, FL, RR, RL
            for legID in range (4):
                J, ee_pos_legFrame = self._robot._ComputeJacobianAndPositionUSC(legID)
                # print( ee_pos_legFrame)

            # foot position at start
            hip_pos = 0.047 # need to check again
            pf_FR = np.array([0.183, -0.083- hip_pos, 1])
            pf_FL= np.array([0.183, 0.083+ hip_pos, 1])
            pf_RR= np.array([-0.183, -0.083- hip_pos, 1])
            pf_RL= np.array([-0.183, 0.083+ hip_pos, 1])
            print("CoM position:", self._robot.GetBasePosition())

            r_FR = pf_FR - np.array(self._robot.GetBasePosition())
            r_FL = pf_FL - np.array(self._robot.GetBasePosition())
            r_RR = pf_RR - np.array(self._robot.GetBasePosition())
            r_RL = pf_RL - np.array(self._robot.GetBasePosition())

            tau_FR = np.cross(r_FR, GRF_FR)
            tau_FL = np.cross(r_FL, GRF_FL)
            tau_RR = np.cross(r_RR, GRF_RR)
            tau_RL = np.cross(r_RL, GRF_RL)
            tau = np.concatenate((tau_FR, tau_FL, tau_RR, tau_RL))
            self._foot_torque.append(tau)

            print("tau_FR:", tau_FR)
            print("tau_FL:", tau_FL)
            print("tau_RR:", tau_RR)
            print("tau_RL:", tau_RL)

            print(" ------------------- ")

            if self._is_render:
                time.sleep(0.001)
                self._render_step_helper()

        print("sim_step_counter:", self._sim_step_counter)
        print("get_sim_time:", self.get_sim_time())

        self._env_step_counter += 1

        # if self.get_sim_time() > 2000 and self._is_render:
        #     time.sleep(0.5)
        #     self._render_step_helper()

        # return {"base_pos_orn": self._base_poss_orns}
        return np.array(self._base_poss_orns), np.array(self._base_vel), np.array(self._foot_forces), np.array(self._foot_torque)

    ######################################################################################

    def get_traj_state_at_time(self, t):
        """ Get the traj state at time t.
        Return:
        -base_pos
        -base_orn
        -base_linvel
        -base_angvel
        -joint_pos
        -joint_vel
        """
        traj_state = self._traj_task.get_state_at_time(t)
        base_pos = self._traj_task.get_base_pos_from_state(traj_state)  # ,env2D=self._env2D)
        base_orn = self._traj_task.get_base_orn_from_state(traj_state)  # ,env2D=self._env2D)
        base_linvel = self._traj_task.get_base_linvel_from_state(traj_state)  # ,env2D=self._env2D)
        base_angvel = self._traj_task.get_base_angvel_from_state(traj_state)  # ,env2D=self._env2D)

        joint_pos = self._traj_task.get_joint_pos_from_state(traj_state)  # ,env2D=self._env2D)
        joint_vel = self._traj_task.get_joint_vel_from_state(traj_state)  # ,env2D=self._env2D)

        return base_pos, base_orn, base_linvel, base_angvel, joint_pos, joint_vel

    def get_traj_state_at_index(self, idx):
        """ Get the traj state at index idx.
        Return:
        -base_pos
        -base_orn
        -base_linvel
        -base_angvel
        -joint_pos
        -joint_vel
        """
        traj_state = self._traj_task.get_state_at_index(idx)
        base_pos = self._traj_task.get_base_pos_from_state(traj_state)  # ,env2D=self._env2D)
        base_orn = self._traj_task.get_base_orn_from_state(traj_state)  # ,env2D=self._env2D)
        base_linvel = self._traj_task.get_base_linvel_from_state(traj_state)  # ,env2D=self._env2D)
        base_angvel = self._traj_task.get_base_angvel_from_state(traj_state)  # ,env2D=self._env2D)

        joint_pos = self._traj_task.get_joint_pos_from_state(traj_state)  # ,env2D=self._env2D)
        joint_vel = self._traj_task.get_joint_vel_from_state(traj_state)  # ,env2D=self._env2D)

        return base_pos, base_orn, base_linvel, base_angvel, joint_pos, joint_vel

    def get_cartesian_traj_state_at_time(self, t):
        """ Get the Cartesian related info at time t. """
        traj_state = self._traj_task.get_state_at_time(t)
        foot_pos = self._traj_task.get_foot_pos_from_state(traj_state)
        foot_vel = self._traj_task.get_foot_vel_from_state(traj_state)
        foot_force = self._traj_task.get_foot_force_from_state(traj_state)

        return foot_pos, foot_vel, foot_force

    def get_robot_state(self):
        """ Get the current robot state.
        Return:
        -base_pos
        -base_orn
        -base_linvel
        -base_angvel
        -joint_pos
        -joint_vel
        """
        base_pos = self._robot.GetBasePosition()
        base_orn = self._robot.GetBaseOrientationRollPitchYaw()
        # print('new base euler', base_orn)
        base_orn = self.convertAnglesToPlusMinusPi2(base_orn)
        # print('last base euler', self._last_base_euler)
        # self._last_base_euler = base_orn
        # print('curr base euler', self._last_base_euler)

        base_linvel = self._robot.GetBaseLinearVelocity()
        # base_angvel = self._robot.GetBaseAngularVelocity()
        base_angvel = self._robot.GetTrueBaseRollPitchYawRate()
        # joint_pos, joint_vel = np.split(self._robot.GetJointState(), 2)
        joint_pos = self._robot.GetMotorAngles()
        joint_vel = self._robot.GetMotorVelocities()

        return base_pos, base_orn, base_linvel, base_angvel, joint_pos, joint_vel

    def convertAnglesToPlusMinusPi2(self, angles):
        """Converts angles to +/- pi/2

        Some lingering bugs in wraparound, and pitch should be double checked as well.
        May be better off keeping everything in quaternions

        """
        base_orn = self._robot.GetBaseOrientation()
        new_angles = angles.copy()
        for i, a in zip([0, 2, 1], [angles[0], angles[2], angles[1]]):  # enumerate(angles):
            if i == 1:
                # check if the value of transforming pitch back to quaternion space is same
                # if not, check value, and can subtract it from pi or -pi
                # print('current quaternion', base_orn)
                # print('pitch from conversion', a)
                new_base_orn = self._pybullet_client.getQuaternionFromEuler([new_angles[0], a, new_angles[2]])
                # print('pitch to quaternion', env._pybullet_client.getQuaternionFromEuler([angles[0], a, angles[2]]))
                if not np.allclose(base_orn, new_base_orn, rtol=1e-3, atol=1e-3):
                    # if np.linalg.norm(np.array(base_orn) - np.array(new_base_orn)) > 5e-7:
                    # if abs(base_orn[1] - new_base_orn[1]) > 1e-4:
                    if a <= 0:
                        new_angles[i] = -np.pi - a
                    else:  # should never get here
                        new_angles[i] = np.pi - a
                else:
                    new_angles[i] = a

            elif abs(self._last_base_euler[i] - a) > np.pi / 2:
                # print('i', i)
                if self._last_base_euler[i] > new_angles[i]:
                    while self._last_base_euler[i] > new_angles[i] and abs(
                            self._last_base_euler[i] - new_angles[i]) > np.pi / 2:
                        # skipped across 0 to around pi
                        # print('pre  update: self._last_base_euler[i]', _last_base_euler[i], 'new_angles[i]',  new_angles[i])
                        new_angles[i] += np.pi
                        # print('post update: self._last_base_euler[i]', _last_base_euler[i], 'new_angles[i]',  new_angles[i])
                elif self._last_base_euler[i] < new_angles[i]:
                    while self._last_base_euler[i] < new_angles[i] and abs(
                            self._last_base_euler[i] - new_angles[i]) > np.pi / 2:
                        # print('pre  update: self._last_base_euler[i]', _last_base_euler[i], 'new_angles[i]',  new_angles[i])
                        new_angles[i] -= np.pi
                        # print('post update: self._last_base_euler[i]', _last_base_euler[i], 'new_angles[i]',  new_angles[i])
            else:
                new_angles[i] = a
        return np.array(new_angles)

        ######################################################################################
        # Render and misc
        ######################################################################################

    def set_env_randomizer(self, env_randomizer):
        self._env_randomizer = env_randomizer

        # def recordVideoHelper(self):
        #   """ Helper to record video, if not already, or end and start a new one """
        #   # If no ID, this is the first video, so make a directory and start logging
        #   if self.videoLogID == None:
        #     #directoryName = VIDEO_LOG_DIRECTORY + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")
        #     directoryName = VIDEO_LOG_DIRECTORY
        #     assert isinstance(directoryName, str)
        #     os.makedirs(directoryName, exist_ok=True)
        #     self.videoDirectory = directoryName
        #   else:
        #     # stop recording and record a new one
        #     self.stopRecordingVideo()

        #   # name
        #   traj_name = self.opt_traj_filenames[self.curr_opt_traj]
        #   output_video_filename = self.videoDirectory + '/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f") + traj_name[:-4] + ".MP4"
        #   logID = self.startRecordingVideo(output_video_filename)
        #   self.videoLogID = logID

    # if parse_version(gym.__version__) < parse_version('0.9.6'):
    #     _render = render
    #     _reset = reset
    #     _seed = seed
    #     _step = step


if __name__ == '__main__':
    NUM_JUMPS = 2
    for i in range (NUM_JUMPS):
        base_poss_orns, base_vel, foot_forces, foot_torques = ImitationGymEnv().jump()
        # print(base_poss)
        # print(obs)
        df1 = pd.DataFrame(np.array(base_poss_orns))
        df1.to_csv("base_poss" + str(i)+ ".csv", index=False, header=False)
        # print('exported data for base_poss for test:', i)

        df2 = pd.DataFrame(np.array(base_vel))
        df2.to_csv("base_vels" + str(i) + ".csv", index = False, header= False)

        df3 = pd.DataFrame(np.array(foot_forces))
        df3.to_csv("foot_forces"  + str(i) + ".csv", index=False, header=False) # Foot force reference does not change for each jump

        df4 = pd.DataFrame(np.array(foot_torques))
        df4.to_csv("foot_torques"  + str(i) + ".csv", index=False, header=False) # Foot force reference does not change for each jump

    print('All Exported done!!')

import os
import math
import pybullet as p
import pybullet_planning as ppu
import pybullet_data
import numpy as np
import gymnasium
from gymnasium import spaces
import time

PWD = os.getcwd()
UR_ROBOT_URDF = os.path.join(PWD, 'universal_robot', 'ur_description', 'urdf', 'ur5.urdf')

"""
State/Observation: robot eef position, target eef position (concatenated, 6d vector)
Action: delta in robot eef position, range from [-1, 1], can be mapped to different physical range e.g. [-0.1m, 0.1m]
"""

## TODO:
## 1. collision check occasionally crashes

class RobotBinPacking(gymnasium.Env):
    def __init__(self):
        super(RobotBinPacking, self).__init__()

        # 初始化PyBullet模拟环境
        ppu.connect(use_gui=True)
        ppu.set_camera_pose(camera_point=[1, -1, 1])
        # p.connect(p.GUI)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.setGravity(0, 0, -9.8)

        # 加载机械手和长方体
        self.robot = ppu.load_pybullet(UR_ROBOT_URDF, fixed_base=True)
        ppu.dump_body(self.robot)
        self.joints = ppu.get_movable_joints(self.robot)
        self.tool_link = ppu.link_from_name(self.robot, 'tool0')
        
        # self.box = ppu.create_box(0.5, 1., 0.8)
        # block_x = 0.5
        # block_y = 1
        # block_z = 0.5
        # ppu.set_pose(self.box, ppu.Pose(ppu.Point(x=block_x, y=block_y, z=block_z), ppu.Euler(yaw=np.pi/2)))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.start_joint_position = np.array([-1.57, -1.5, 1.6, 1.57, 1.57, 0.])
        ppu.set_joint_positions(self.robot, self.joints, self.start_joint_position)
        self.des_next_wp = self.get_end_effector_position()
        print("Initial position: ", self.des_next_wp)
        self.target_position=np.array([0.2,0.3,0.5]) # will be passed in later

        # collision
        robot_self_collision_disabled_link_names = [('base_link', 'shoulder_link'),
        ('ee_link', 'wrist_1_link'), ('ee_link', 'wrist_2_link'),
        ('ee_link', 'wrist_3_link'), 
        ('tool0', 'wrist_1_link'), ('tool0', 'wrist_2_link'),
        ('tool0', 'wrist_3_link'), ('forearm_link', 'upper_arm_link'),
        ('forearm_link', 'wrist_1_link'), ('shoulder_link', 'upper_arm_link'),
        ('wrist_1_link', 'wrist_2_link'), ('wrist_1_link', 'wrist_3_link'),
        ('wrist_2_link', 'wrist_3_link')]
        self.self_collision_links = ppu.get_disabled_collisions(self.robot, robot_self_collision_disabled_link_names)

        # 环境参数
        total_step = 100
        self.max_steps = total_step
        self.current_step = 0


    def get_observation(self):
        # use desired waypoint position, for now
        robot_state = self.des_next_wp
        return np.concatenate((robot_state, self.target_position)).flatten()
    
    def get_observed_eef_position(self, obs):
        return obs[:3]

    def get_mapped_action(self, normalized_action):
        # assumes delta position in physical world is 0.1m in each dimension
        return normalized_action * 0.05

    def step(self, action):
        mapped_action = self.get_mapped_action(action)
        next_waypoint = self.des_next_wp + mapped_action

        # Get the current state
        state = self.get_observation()

        success = self.move_to_next_waypoint(next_waypoint)

        # Calculate the reward
        reward = self.calculate_reward(state, action, success)

        # Perform a simulation step
        p.stepSimulation()

        # Check if the task is done
        done = self.is_done(state)
        truncated = False
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            truncated = True
        # with open('rewards.csv', 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([reward])
        info = {}

        # remember the desired next waypoint
        self.des_next_wp = next_waypoint

        return state, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        # Reset the PyBullet simulation
        # ppu.reset_simulation()
        # p.setGravity(0, 0, -9.8)

        # # Reload the robot and set its initial position
        # self.robot = ppu.load_pybullet(UR_ROBOT_URDF, fixed_base=True)
        # self.joints = ppu.get_movable_joints(self.robot)
        # self.tool_link = ppu.link_from_name(self.robot, 'tool0')

        # Reset the robot to initial joint position
        ppu.set_joint_positions(self.robot, self.joints, self.start_joint_position)
        self.des_next_wp = self.get_end_effector_position()

        # Reset the environment parameters
        self.current_step = 0

        # Return the initial observation

        return self.get_observation(), {}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()



    def is_done(self, state):
        threshold = 0.001
        distance = np.linalg.norm(self.get_observed_eef_position(state) - self.target_position)
        return distance < threshold

    def get_end_effector_position(self):
        end_effector_link_index = 6 #assuming it is 6
        # 获取末端执行器的状态
        state = p.getLinkState(self.robot, end_effector_link_index)
        position = state[0]
        return np.concatenate([position])

    # then store collision function and check in reward calculation
    def move_to_next_waypoint(self, next_waypoint):
        end_pose = ppu.Pose(ppu.Point(next_waypoint[0], next_waypoint[1], next_waypoint[2]), ppu.Euler(roll=np.pi))
        end_conf = ppu.inverse_kinematics_helper(self.robot, self.tool_link, end_pose)
        if end_conf is None:
            return False
        jt_path = ppu.plan_joint_motion(self.robot, self.joints, end_conf, resolutions=0.05*np.ones(len(self.joints)),
                                        self_collisions=True, disabled_collisions=self.self_collision_links)
        if jt_path is None:
            print("Motio planning failure")
            return False
        return True

    def calculate_reward(self, state, action, motion_planner_success):
        # calculate the distance between current end effector (x,y,z) and target end effector position(x,y,z)
        # x1,y1,z1 = self.get_observed_eef_position(state)
        # x2,y2,z2 = self.target_position
        # distance=math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

        distance = np.linalg.norm(self.get_observed_eef_position(state) - self.target_position, ord=np.inf)

        # reward = -1*distance + (0 if motion_planner_success else -5)
        # reward = -5*(float(self.current_step) / float(self.max_steps))*distance + (0 if motion_planner_success else -5)
        reward = -5*(np.sqrt(float(self.current_step) / float(self.max_steps)))*distance + (0 if motion_planner_success else -5)
        # # collision penalty
        # collision_fn = ppu.get_collision_fn(self.robot, self.joints, obstacles=[],
        #                                  self_collisions=True)
        # pose = ppu.Pose(ppu.Point(x1, y1, z1))
        # jt_pos = ppu.inverse_kinematics_helper(self.robot, self.tool_link, pose)
        # collision_penalty = -1 if collision_fn(jt_pos) else 0
        # reward += collision_penalty

        # print("state:", x1, y1, z1)
        # print("action: ", action)
        # print("reward: ", reward)

        return reward
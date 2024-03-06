import numpy as np
import pybullet_planning as ppu
from stable_baselines3 import SAC

import os
import time

PWD = os.getcwd()
UR_ROBOT_URDF = os.path.join(PWD, 'universal_robot', 'ur_description', 'urdf', 'ur5.urdf')

# return RL model
def run_rl(env):
  model = SAC("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=10000, log_interval=4)
  model.save("sac_waypoints")

  return model



def main():
  # model = run_rl()

  # obs, info = env.reset()

  # pybullet 
  ppu.connect(use_gui=True)
  ppu.set_camera_pose(camera_point=[1, -1, 1])
  robot = ppu.load_pybullet(UR_ROBOT_URDF, fixed_base=True)
  ppu.dump_body(robot)
  tool_link = ppu.link_from_name(robot, 'tool0')
  joints = ppu.get_movable_joints(robot)
  ppu.set_joint_positions(robot, joints, np.array([-1.57, -1.5, 1.6, 1.57, 1.57, 0.]))
  curr_pose = ppu.get_link_pose(robot, tool_link)
  print(curr_pose)
  ppu.wait_for_user()


  while True:
    # create box
    # block = ppu.create_box(0.5, 1., 0.8)
    # block_x = 0.5
    # block_y = 1
    # block_z = 0.5
    # ppu.set_pose(block, ppu.Pose(ppu.Point(x=block_x, y=block_y, z=block_z), ppu.Euler(yaw=np.pi/2)))

    # action, _states = model.predict(obs, deterministic=True)
    # obs, reward, terminated, truncated, info = env.step(action)
    # next_waypoint = action

    # motion plan to next_waypoint
    next_waypoint = [0.4, 0.4, 0.4]
    end_pose = ppu.Pose(ppu.Point(next_waypoint[0], next_waypoint[1], next_waypoint[2]))
    end_conf = ppu.inverse_kinematics_helper(robot, tool_link, end_pose)
    # end_conf = np.array([ 0.59131901, -0.88109483,  1.13689071,  1.31499328, -1.57080056,  0.97947445])
    if end_conf is None:
      print("IK failure")
      ppu.disconnect()
      return
    curr_pose = ppu.get_joint_positions(robot, joints)
    jt_path = ppu.plan_joint_motion(robot, joints, end_conf, resolutions=0.05*np.ones(len(joints)))
    print(jt_path)
    ppu.set_joint_positions(robot, joints, curr_pose)
    ppu.wait_for_user()
    if jt_path is None:
      print('Path planning failure!')
      ppu.disconnect()
      return

    # inner loop to move robot to next_waypoint
    for conf in jt_path:
      ppu.set_joint_positions(robot, joints, conf)
      ppu.wait_for_duration(0.1)
    break
    
    # if terminated or truncated:
    #   obs, info = env.reset()
    #   break
    



  ppu.wait_for_duration(5)
  ppu.disconnect()

  

if __name__ == "__main__":
  main()
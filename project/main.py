import numpy as np
import pybullet_planning as ppu
import gymnasium 
from stable_baselines3.common.monitor import Monitor

from env import RobotBinPacking
from rl import run_rl, LOG_DIR

import os
import time


def main():
  env = RobotBinPacking()
  env = Monitor(env, LOG_DIR)

  # # quick test
  # obs, reward, terminated, info = env.step(np.array([0.7, 0, 0.1]))

  model = run_rl(env)
  ppu.wait_for_user()

  obs, info = env.reset()

  step = 0
  while True:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, terminated, truncated, info = env.step(action)
      env.render()
      if step % 5:
        print(env.get_end_effector_position())
      step += 1
      if terminated:
          break

  print("Final position: ", env.get_end_effector_position())
  ppu.wait_for_duration(5)
  ppu.disconnect()

  

if __name__ == "__main__":
  main()
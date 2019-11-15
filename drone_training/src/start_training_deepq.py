#!/usr/bin/env python

import gym
import time
import numpy
import random
import time
from baselines import deepq

from gym import wrappers

# ROS packages required
import rospy
import rospkg
import os

# import our training environment
#import turtlebot2_maze
from openai_ros.task_envs.turtlebot2 import turtlebot2_maze


if __name__ == '__main__':
    
    rospy.init_node('turtlebot_gym', anonymous=True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('turtlebot-v0')
    print("Here is the action space")
    print(env.action_space)
    rospy.loginfo ( "Gym environment done")
        
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('drone_training')
    models_dir_path = os.path.join(pkg_path, "models_saved")
    if not os.path.exists(models_dir_path):
        os.makedirs(models_dir_path)
    
    out_model_file_path = os.path.join(models_dir_path, "movingcube_model.pkl")
    
    rospy.loginfo ( "Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    max_timesteps = rospy.get_param("/turtlebot2/max_timesteps")
    buffer_size = rospy.get_param("/turtlebot2/buffer_size")
    # We convert to float becase if we are using Ye-X notation, it sometimes treats it like a string.
    lr = float(rospy.get_param("/turtlebot2/lr"))
    
    exploration_fraction = rospy.get_param("/turtlebot2/exploration_fraction")
    exploration_final_eps = rospy.get_param("/turtlebot2/exploration_final_eps")
    print_freq = rospy.get_param("/turtlebot2/print_freq")
    
    reward_task_learned = rospy.get_param("/turtlebot2/reward_task_learned")


    def callback(lcl, _glb):
        # stop training if reward exceeds 199
        aux1 = lcl['t'] > 100
        aux2 = sum(lcl['episode_rewards'][-101:-1]) / 100
        is_solved = aux1 and aux2 >= reward_task_learned
        
        rospy.logdebug("aux1="+str(aux1))
        rospy.logdebug("aux2="+str(aux2))
        rospy.logdebug("reward_task_learned="+str(reward_task_learned))
        rospy.logdebug("IS SOLVED?="+str(is_solved))
        
        return is_solved

    # Initialises the algorithm that we are going to use for learning
    act = deepq.learn(
        env,
        network='cnn',
        lr=lr,
        total_timesteps=max_timesteps, 
        buffer_size=buffer_size,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        print_freq=print_freq, # how many apisodes until you print total rewards and info
        param_noise=False,
        callback=callback
    )
    env.close()
    rospy.logwarn("Saving model to movingcube_model.pkl")
    act.save(out_model_file_path)



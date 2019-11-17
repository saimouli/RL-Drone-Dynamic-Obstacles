# RL-Drone-Dynamic-Obstacles
Research project involving RL for the dynamic obstacle avoidance using AR drone 2.0 platform 

# TO-DO
- [ ] Integrate intel real sense sensor into the ar drone urdf
- [ ] Install openAI ROS gazebo  
- [ ] Integrate openAI and ar drone 

## Build instructions 

1. Clone the package into your catkin workspace
2. We need to use python3 for reinforcement learning packages, but ROS is currently supported for Python2. We need build TF packages additionally to successfully run the code. You might get following errors if you do not build TF packages for python 3.
'ImportError: dynamic module does not define module export function (PyInit__tf2)'

3. Now once you have cloned the package build it using following command.
`catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so`
You might get following error message while compiling the package.
'fatal error: Python.h: No such file or directory'
This is because you might not be using python3.6, so you have to find the version of the python and then replace the python version in the catkin_make command stated above. For me it is python3.5. 
* Refer following links for more info.
 https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674
 https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/

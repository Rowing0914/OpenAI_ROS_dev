sudo pip install gym==0.8.0
sudo pip install tensorflow
sudo pip uninstall numpy
sudo pip install numpy
sudo pip install python-git

sudo apt install ros-kinetic-moveit-commander
sudo apt install ros-kinetic-opencv-candidate
sudo apt install ros-kinetic-costmap_2d
sudo apt install ros-kinetic-robot-controllers
sudo apt install ros-kinetic-rgbd-launch
sudo apt install ros-kinetic-ompl
sudo apt install ros-kinetic-moveit-simple-controller-manager
# You might need some other packages from ros, you will see when compiling

# Chancge The Path to your workspace in the file:
# fetch_openai_ros_example fetch_n1try_params_v2.yaml
# ros_ws_abspath: "/PATH_TO_YOUR_CATKIN_WS/catkin_ws"


mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_make
cd ~/catkin_ws/src
git clone https://RDaneelOlivaw@bitbucket.org/theconstructcore/openai_ros.git
cd openai_ros
git checkout version2
cd ~/catkin_ws/src
git clone https://RDaneelOlivaw@bitbucket.org/theconstructcore/openai_examples_projects.git
cd ~/catkin_ws
catkin_make
source devel/setup.bash
rospack profile

# Launch the Training 
roslaunch fetch_openai_ros_example start_training_n1try_v2_save_and_load.launch
# Fist time you launch it it will start downloading the gits for the simulations
# Just let it finish until a message appears stating that you have to compile.
# Then just CTRL+C, follow the instructions given for compiling and launch again.

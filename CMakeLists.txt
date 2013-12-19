cmake_minimum_required(VERSION 2.8.3)
project(template_pose_estimation)
find_package(catkin REQUIRED COMPONENTS 
	roscpp
	sensor_msgs
	image_transport
	image_geometry
	cv_bridge
	tf
	nav_msgs
	message_filters)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS signals filesystem system)

catkin_package()

include_directories(${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} include)

add_executable(template_pose 
				src/template_pose_node.cpp
				src/template_pose.cpp
				src/image_properties.cpp)
target_link_libraries(template_pose ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})
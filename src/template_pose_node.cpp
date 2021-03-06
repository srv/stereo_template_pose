/**
 * @file
 * @brief ROS node for stereo_template_pose code
 */


#include <ros/ros.h>
#include "template_pose.h"

int main(int argc, char **argv)
{
  ros::init(argc,argv,"template_pose");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  template_pose::TemplatePose template_pose(nh,nh_private);
  ros::spin();
  return 0;
}
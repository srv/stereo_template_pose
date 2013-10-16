#ifndef TEMPLATE_POSE_BASE_H
#define TEMPLATE_POSE_BASE_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>
#include <image_geometry/stereo_camera_model.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>
#include "opencv2/core/core.hpp"
#include "image_properties.h"

using namespace std;
using namespace cv;

namespace enc = sensor_msgs::image_encodings;

namespace template_pose
{

class TemplatePoseBase
{

public:

	// Constructor
  TemplatePoseBase(ros::NodeHandle nh, ros::NodeHandle nhp);

  struct Params
  {
    //Default constructor sets all values to defaults.
    Params();

    // Class parameters
    int queue_size;                   //!> Indicate the maximum number of messages encued.
    double desc_threshold;            //!> Descriptor threshold (0-1).
    string template_image_name;       //!> Name of the template image located at the node path into '/etc' directory (example: template.png)
    int min_matches;                  //!> Minimum number of matches to estimate the transform between template and stereo frame.
    int min_inliers;                  //!> Minimum number of inliers to considerate the transform valid.
    string stereo_frame_id;           //!> Frame id for the stereo camera.
    string template_frame_id;         //!> Frame id for the template image.

    // Default values
    static const int          DEFAULT_QUEUE_SIZE = 1;
    static const double       DEFAULT_DESC_THRESHOLD = 0.8;
    static const int          DEFAULT_MIN_MATCHES = 70;
    static const int          DEFAULT_MIN_INLIERS = 25;
  };

  // Set the parameter struct
  inline void setParams(const Params& params) { params_ = params; }

  // Return current parameters
  inline Params params() const { return params_; }

protected:

	// Node handlers
	ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  bool initialize();
  void readParameters();
  void msgsCallback(const sensor_msgs::ImageConstPtr& l_img,
                    const sensor_msgs::ImageConstPtr& r_img,
                    const sensor_msgs::CameraInfoConstPtr& l_info,
                    const sensor_msgs::CameraInfoConstPtr& r_info);

private:

  // Estimates the output trasnform between template and stereo frame
  bool estimateTransform(tf::Transform& output);

  // Topic properties
  image_transport::SubscriberFilter left_sub_, right_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> left_info_sub_, right_info_sub_;

  // Topic sync properties
  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, 
                                                    sensor_msgs::Image, 
                                                    sensor_msgs::CameraInfo, 
                                                    sensor_msgs::CameraInfo> ExactPolicy;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                                    sensor_msgs::Image, 
                                                    sensor_msgs::CameraInfo, 
                                                    sensor_msgs::CameraInfo> ApproximatePolicy;
  typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
  typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
  boost::shared_ptr<ExactSync> exact_sync_;
  boost::shared_ptr<ApproximateSync> approximate_sync_;

  // Stores parameters
  Params params_;

  // Operational parameters
  tf::TransformBroadcaster tf_broadcaster_;   //!> Transform publisher
  ImageProperties template_prop_;             //!> Image properties for the template image
  ImageProperties stereo_prop_;               //!> Image properties for the stereo frame
  Mat camera_matrix_;                         //!> Camera matrix
  bool first_iter_;                           //!> True in the first iteration
};

} // namespace

#endif // TEMPLATE_POSE_BASE_H
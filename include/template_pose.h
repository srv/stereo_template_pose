#ifndef TEMPLATE_POSE_H
#define TEMPLATE_POSE_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>
#include <std_srvs/Empty.h>
#include "opencv2/core/core.hpp"
#include "image_properties.h"

using namespace std;
using namespace cv;

namespace enc = sensor_msgs::image_encodings;

namespace template_pose
{

class TemplatePose
{

public:

	// Constructor
  TemplatePose(ros::NodeHandle nh, ros::NodeHandle nhp);

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
    string frame_id;                  //!> Frame id for the stereo camera.
    string template_frame_id;         //!> Frame id for the template image.
    string desc_type;                 //!> Descriptor type can be: SIFT or SURF.
    int bucket_width;                 //!> Bucket width.
    int bucket_height;                //!> Bucket height.
    int max_bucket_features;          //!> Maximum number the features per bucket.
    double template_width;            //!> The template image width in meters.
    double template_height;           //!> The template image height in meters.

    // Default values
    static const int            DEFAULT_QUEUE_SIZE = 1;
    static const double         DEFAULT_DESC_THRESHOLD = 0.8;
    static const int            DEFAULT_MIN_MATCHES = 70;
    static const int            DEFAULT_MIN_INLIERS = 25;
    static const int            DEFAULT_BUCKET_WIDTH = 30;
    static const int            DEFAULT_BUCKET_HEIGHT = 30;
    static const int            DEFAULT_MAX_BUCKET_FEATURES = 10;
    static const double         DEFAULT_TEMPLATE_WIDTH = 0.5;
    static const double         DEFAULT_TEMPLATE_HEIGHT = 0.16;
  };

  // Set the parameter struct
  inline void setParams(const Params& params) { params_ = params; }

  // Return current parameters
  inline Params params() const { return params_; }

protected:

	// Node handlers
	ros::NodeHandle nh_;
  ros::NodeHandle nhp_;

  bool initialize();
  void readParameters();
  void msgsCallback(const sensor_msgs::ImageConstPtr& img,
                    const sensor_msgs::CameraInfoConstPtr& img_info);

private:

  // Estimates the output trasnform between template and stereo frame
  bool estimateTransform(tf::Transform& output);

  // Topic properties
  image_transport::SubscriberFilter image_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub_;
  ros::Publisher matches_image_pub_;

  // Services
  ros::ServiceServer detect_service_;
  ros::ServiceServer start_service_;
  ros::ServiceServer stop_service_;
  bool listen_services_;
  bool do_detection_;
  bool toggle_detection_; // just one detection is required
  bool detectSrv(std_srvs::Empty::Request&, std_srvs::Empty::Response&);
  bool startDetectionSrv(std_srvs::Empty::Request&, std_srvs::Empty::Response&);
  bool stopDetectionSrv(std_srvs::Empty::Request&, std_srvs::Empty::Response&);


  // Topic sync properties
  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, 
                                                    sensor_msgs::CameraInfo> ExactPolicy;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                                    sensor_msgs::CameraInfo> ApproximatePolicy;
  typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
  typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
  boost::shared_ptr<ExactSync> exact_sync_;
  boost::shared_ptr<ApproximateSync> approximate_sync_;

  // Stores parameters
  Params params_;

  // Operational parameters
  tf::Transform camera_to_template_;          //!> Transform from camera to template
  tf::TransformBroadcaster tf_broadcaster_;   //!> Transform publisher
  ImageProperties template_prop_;             //!> Image properties for the template image
  ImageProperties img_prop_;               //!> Image properties for the stereo frame
  Mat camera_matrix_;                         //!> Camera matrix
  bool first_iter_;                           //!> True in the first iteration
};

} // namespace

#endif // TEMPLATE_POSE_H
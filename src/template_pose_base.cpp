#include <ros/package.h>
#include "template_pose_base.h"
#include <boost/shared_ptr.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include "opencv_utils.h"

/** \brief Parameter constructor. Sets the parameter struct to default values.
  */
template_pose::TemplatePoseBase::Params::Params() :
  queue_size(DEFAULT_QUEUE_SIZE),
  desc_threshold(DEFAULT_DESC_THRESHOLD),
  min_matches(DEFAULT_MIN_MATCHES),
  min_inliers(DEFAULT_MIN_INLIERS)
{}

/** \brief Class constructor. Reads node parameters and initialize some properties.
  * \param nh public node handler
  * \param nhp private node handler
  */
template_pose::TemplatePoseBase::TemplatePoseBase(
  ros::NodeHandle nh, ros::NodeHandle nhp) : nh_(nh), nh_private_(nhp)
{
  // Read the node parameters
  readParameters();

  // Initialize the stereo slam
  initialize();
}

/** \brief Messages callback. This function is called when synchronized image
  * messages are received.
  * @return 
  * \param l_img ros image message of type sensor_msgs::Image
  * \param r_img ros image message of type sensor_msgs::Image
  * \param l_info ros info image message of type sensor_msgs::CameraInfo
  * \param r_info ros info image message of type sensor_msgs::CameraInfo
  */
void template_pose::TemplatePoseBase::msgsCallback(
                                  const sensor_msgs::ImageConstPtr& l_img,
                                  const sensor_msgs::ImageConstPtr& r_img,
                                  const sensor_msgs::CameraInfoConstPtr& l_info,
                                  const sensor_msgs::CameraInfoConstPtr& r_info)
{
  // Convert image message to Mat
  cv_bridge::CvImagePtr l_ptr, r_ptr;
  try
  {
    l_ptr = cv_bridge::toCvCopy(l_img, enc::BGR8);
    r_ptr = cv_bridge::toCvCopy(r_img, enc::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("[StereoSlam:] cv_bridge exception: %s", e.what());
    return;
  }

  if(first_iter_)
  {
    // Initialize camera matrix    
    image_geometry::StereoCameraModel stereo_camera_model;
    stereo_camera_model.fromCameraInfo(l_info, r_info);
    const Mat P(3,4, CV_64FC1, const_cast<double*>(l_info->P.data()));
    camera_matrix_ = P.colRange(Range(0,3)).clone();

    // Set the stereo model for the stereo properties
    stereo_prop_.setCameraModel(stereo_camera_model);

    // Do not repeat
    first_iter_ = false;
  }

  // Set the images
  stereo_prop_.setLeftImg(l_ptr->image);
  stereo_prop_.setRightImg(r_ptr->image);

  // Estimate the transform
  tf::Transform est_tf;
  if(estimateTransform(est_tf))
    camera_to_template_ = est_tf;
  else
    ROS_INFO("[TemplatePose:] No transform found.");

  // Publish the transform
  tf_broadcaster_.sendTransform(
    tf::StampedTransform(camera_to_template_, l_img->header.stamp,
    params_.stereo_frame_id, params_.template_frame_id));

  // Log
  double x, y, z, roll, pitch, yaw;
  camera_to_template_.getBasis().getRPY(roll, pitch, yaw);
  x = camera_to_template_.getOrigin().x();
  y = camera_to_template_.getOrigin().y();
  z = camera_to_template_.getOrigin().z();
  ROS_INFO_STREAM("[TemplatePose:] Camera to template image:" <<
                  " [" << x << ", " << y << ", " << z << 
                  ", " << roll*180/M_PI << ", " << pitch*180/M_PI << 
                  ", " << yaw*180/M_PI << "]");
}

/** \brief Reads the stereo slam node parameters
  */
void template_pose::TemplatePoseBase::readParameters()
{
  Params template_pose_params;

  // General parameters
  nh_private_.getParam("queue_size", template_pose_params.queue_size);
  nh_private_.getParam("desc_threshold", template_pose_params.desc_threshold);
  nh_private_.getParam("min_matches", template_pose_params.min_matches);
  nh_private_.getParam("min_inliers", template_pose_params.min_inliers);
  nh_private_.param("template_image_name", template_pose_params.template_image_name, std::string("template.png"));
  nh_private_.param("stereo_frame_id", template_pose_params.stereo_frame_id, std::string("/stereo_down"));
  nh_private_.param("template_frame_id", template_pose_params.template_frame_id, std::string("/template_image"));

  // Set stereo slam parameters
  setParams(template_pose_params);

  // Topics subscriptions
  string left_topic, right_topic, left_info_topic, right_info_topic;
  nh_private_.param("left_topic", left_topic, string("/left/image_rect_color"));
  nh_private_.param("right_topic", right_topic, string("/right/image_rect_color"));
  nh_private_.param("left_info_topic", left_info_topic, string("/left/camera_info"));
  nh_private_.param("right_info_topic", right_info_topic, string("/right/camera_info"));
  image_transport::ImageTransport it(nh_);
  left_sub_ .subscribe(it, left_topic, 1);
  right_sub_.subscribe(it, right_topic, 1);
  left_info_sub_.subscribe(nh_, left_info_topic, 1);
  right_info_sub_.subscribe(nh_, right_info_topic, 1);
}

/** \brief Initializes the template pose class
  * @return true if all ok
  */
bool template_pose::TemplatePoseBase::initialize()
{
  // Initialize parameters
  first_iter_ = true;
  camera_to_template_.setIdentity();

  // Initialize image properties objects
  template_pose::ImageProperties::Params image_params;
  image_params.desc_type = "SIFT";
  image_params.desc_threshold = params_.desc_threshold;
  image_params.epipolar_threshold = 3.5;
  template_prop_.setParams(image_params);
  stereo_prop_.setParams(image_params);

  // Check if template image exists
  string template_file =  ros::package::getPath(ROS_PACKAGE_NAME) + 
                          string("/etc/") + params_.template_image_name;
  if (!boost::filesystem::exists(template_file))
  {
    ROS_ERROR_STREAM("[TemplatePose:] The template image file does not exists: " << 
                     template_file);
    return false;
  }

  // Read the template image and extract kp and descriptors
  Mat img_temp = imread(template_file, CV_LOAD_IMAGE_COLOR);
  template_prop_.setLeftImg(img_temp);

  // Callback synchronization
  bool approx;
  nh_private_.param("approximate_sync", approx, false);
  if (approx)
  {
    approximate_sync_.reset(new ApproximateSync(ApproximatePolicy(params_.queue_size),
                                    left_sub_, 
                                    right_sub_, 
                                    left_info_sub_, 
                                    right_info_sub_) );
    approximate_sync_->registerCallback(boost::bind(
        &template_pose::TemplatePoseBase::msgsCallback,
        this, _1, _2, _3, _4));
  }
  else
  {
    exact_sync_.reset(new ExactSync(ExactPolicy(params_.queue_size),
                                    left_sub_, 
                                    right_sub_, 
                                    left_info_sub_, 
                                    right_info_sub_) );
    exact_sync_->registerCallback(boost::bind(
        &template_pose::TemplatePoseBase::msgsCallback, 
        this, _1, _2, _3, _4));
  }

  return true;
}

/** \brief Estimates the transform between the template and the stereo frame
  * @return true if all ok
  * \param - output tf transform
  */
bool template_pose::TemplatePoseBase::estimateTransform(tf::Transform& output)
{
  // Initialize output
  output.setIdentity();

  // Crosscheck descriptors matching
  Mat desc_stereo = stereo_prop_.getLeftDesc3D();
  Mat desc_template = template_prop_.getLeftDesc();
  vector<DMatch> matches;
  Mat match_mask;
  template_pose::OpencvUtils::crossCheckThresholdMatching(desc_stereo,
  desc_template, params_.desc_threshold, match_mask, matches);

  ROS_INFO_STREAM("[TemplatePose:] Found " << matches.size() <<
   " matches (min_matches is: " << params_.min_matches << ")");

  // Sanity check
  if ((int)matches.size() < params_.min_matches)
    return false;

  // Extract keypoints and 3d points of vertex i and j
  vector<KeyPoint> kp_template = template_prop_.getLeftKp();
  vector<Point3f> points3d_stereo = stereo_prop_.get3Dpoints();

  vector<Point2f> matched_keypoints;
  vector<Point3f> matched_3d_points;
  for (size_t i = 0; i < matches.size(); ++i)
  {
    int idx_stereo = matches[i].queryIdx;
    int idx_template = matches[i].trainIdx;
    matched_3d_points.push_back(points3d_stereo[idx_stereo]);
    matched_keypoints.push_back(kp_template[idx_template].pt);
  }
  
  // Compute the transformation between the vertices
  Mat rvec, tvec;
  vector<int> inliers;
  solvePnPRansac(matched_3d_points, matched_keypoints, camera_matrix_, 
                 Mat(), rvec, tvec, false /* no extrinsic guess */,
                 50 /* iterations */, 8.0 /* reproj. error */,
                 params_.min_inliers /* min inliers */, inliers);

  ROS_INFO_STREAM("[TemplatePose:] Found " << inliers.size() <<
   " inliers (min_inliers is: " << params_.min_inliers << ")");

  // Sanity check
  if (static_cast<int>(inliers.size()) < params_.min_inliers)
    return false;

  // Ok, build the tf
  tf::Vector3 axis(rvec.at<double>(0, 0), 
                   rvec.at<double>(1, 0), 
                   rvec.at<double>(2, 0));
  double angle = cv::norm(rvec);
  tf::Quaternion quaternion(axis, angle);

  tf::Vector3 translation(tvec.at<double>(0, 0), tvec.at<double>(1, 0), 
      tvec.at<double>(2, 0));

  output = (tf::Transform(quaternion, translation)).inverse();
  return true;
}
#include <ros/package.h>
#include <cv_bridge/cv_bridge.h>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

#include "opencv_utils.h"
#include "template_pose.h"

/** \brief Parameter constructor. Sets the parameter struct to default values.
  */
template_pose::TemplatePose::Params::Params() :
  queue_size(DEFAULT_QUEUE_SIZE),
  desc_threshold(DEFAULT_DESC_THRESHOLD),
  min_matches(DEFAULT_MIN_MATCHES),
  min_inliers(DEFAULT_MIN_INLIERS),
  desc_type("SIFT"),
  bucket_width(DEFAULT_BUCKET_WIDTH),
  bucket_height(DEFAULT_BUCKET_HEIGHT),
  max_bucket_features(DEFAULT_MAX_BUCKET_FEATURES),
  template_width(DEFAULT_TEMPLATE_WIDTH),
  template_height(DEFAULT_TEMPLATE_HEIGHT)
{}

/** \brief Class constructor. Reads node parameters and initialize some properties.
  * \param nh public node handler
  * \param nhp private node handler
  */
template_pose::TemplatePose::TemplatePose(
  ros::NodeHandle nh, ros::NodeHandle nhp) : nh_(nh), nhp_(nhp)
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
  * \param l_info ros info image message of type sensor_msgs::CameraInfo
  */
void template_pose::TemplatePose::msgsCallback(
                                  const sensor_msgs::ImageConstPtr& img,
                                  const sensor_msgs::CameraInfoConstPtr& img_info)
{

  // Check if service is called or not
  if (listen_services_ && !(do_detection_ || toggle_detection_))
  {
    ROS_INFO_ONCE("[TemplatePose:] Waiting for start service...");
    return;
  }

  // Convert image message to Mat
  cv_bridge::CvImagePtr img_ptr;
  try
  {
    img_ptr = cv_bridge::toCvCopy(img, enc::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("[TemplatePose:] cv_bridge exception: %s", e.what());
    return;
  }

  if(first_iter_)
  {
    // Initialize camera matrix    
    const Mat P(3,4, CV_64FC1, const_cast<double*>(img_info->P.data()));
    camera_matrix_ = P.colRange(Range(0,3)).clone();

    // Do not repeat
    first_iter_ = false;
  }

  // Set the images
  img_prop_.setImg(img_ptr->image);

  // Estimate the transform
  tf::Transform est_tf;
  if(estimateTransform(est_tf))
    camera_to_template_ = est_tf;
  else
    ROS_INFO("[TemplatePose:] No transform found.");

  // Publish the transform
  tf_broadcaster_.sendTransform(
    tf::StampedTransform(camera_to_template_, img->header.stamp,
    params_.frame_id, params_.template_frame_id));

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

  toggle_detection_ = false;
}

/** \brief Reads the stereo slam node parameters
  */
void template_pose::TemplatePose::readParameters()
{
  Params params;

  // General parameters
  nhp_.param("queue_size", params.queue_size, 2);
  nhp_.param("desc_threshold", params.desc_threshold, 0.8);
  nhp_.param("min_matches", params.min_matches, 40);
  nhp_.param("min_inliers", params.min_inliers, 20);
  nhp_.param("template_image_name", params.template_image_name, std::string("template.png"));
  nhp_.param("frame_id", params.frame_id, std::string("/camera"));
  nhp_.param("template_frame_id", params.template_frame_id, std::string("/template_image"));
  nhp_.param("desc_type", params.desc_type, std::string("SIFT"));
  nhp_.param("bucket_width", params.bucket_width, 30);
  nhp_.param("bucket_height", params.bucket_height, 30);
  nhp_.param("max_bucket_features", params.max_bucket_features, 10);
  nhp_.param("template_width", params.template_width, 0.5);
  nhp_.param("template_height", params.template_height, 0.16);

  // Set stereo slam parameters
  setParams(params);

  // Topics subscriptions
  string image_topic, image_info_topic;
  nhp_.param("image_topic", image_topic, string("/left/image_rect_color"));
  nhp_.param("image_info_topic", image_info_topic, string("/left/camera_info"));
  image_transport::ImageTransport it(nh_);
  image_sub_.subscribe(it, image_topic, 1);
  info_sub_.subscribe(nh_, image_info_topic, 1);

  // Services to start or stop the template detection
  nhp_.param("listen_services", listen_services_, false);
  detect_service_ = nhp_.advertiseService("detect", &TemplatePose::detectSrv, this);
  start_service_ = nhp_.advertiseService("start_detection", &TemplatePose::startDetectionSrv, this);
  stop_service_ = nhp_.advertiseService("stop_detection", &TemplatePose::stopDetectionSrv, this);

  // Advertise the image matches publisher
  matches_image_pub_ = nhp_.advertise<sensor_msgs::Image>("matches", 1);
}

/** \brief Initializes the template pose class
  * @return true if all ok
  */
bool template_pose::TemplatePose::initialize()
{
  // Initialize parameters
  first_iter_ = true;
  camera_to_template_.setIdentity();

  // Initialize service bool variables
  if (listen_services_)
    do_detection_ = false;
  else
    do_detection_ = true;

  // Check if template image exists
  string template_file = params_.template_image_name;
  if (!boost::filesystem::exists(template_file))
  {
    ROS_ERROR_STREAM("[TemplatePose:] The template image file does not exists: " << 
                     template_file);
    return false;
  }

  // Read the template image
  Mat img_temp = imread(template_file, CV_LOAD_IMAGE_COLOR);

  // Initialize image properties objects
  template_pose::ImageProperties::Params image_params;
  image_params.desc_type = params_.desc_type;
  image_params.bucket_width = params_.bucket_width;
  image_params.bucket_height = params_.bucket_height;
  image_params.max_bucket_features = params_.max_bucket_features;
  image_params.px_meter_x = (float)img_temp.cols / (float)params_.template_width;
  image_params.px_meter_y = (float)img_temp.rows / (float)params_.template_height;

  // Set the image properties
  template_prop_.setParams(image_params);
  img_prop_.setParams(image_params);

  // Compute template properties
  template_prop_.setImg(img_temp);
  template_prop_.compute3D();

  // Callback synchronization
  bool approx;
  nhp_.param("approximate_sync", approx, false);
  if (approx)
  {
    approximate_sync_.reset(new ApproximateSync(ApproximatePolicy(params_.queue_size),
                                    image_sub_,
                                    info_sub_) );
    approximate_sync_->registerCallback(boost::bind(
        &template_pose::TemplatePose::msgsCallback,
        this, _1, _2));
  }
  else
  {
    exact_sync_.reset(new ExactSync(ExactPolicy(params_.queue_size),
                                    image_sub_,
                                    info_sub_) );
    exact_sync_->registerCallback(boost::bind(
        &template_pose::TemplatePose::msgsCallback, 
        this, _1, _2));
  }

  return true;
}

/** \brief Estimates the transform between the template and the stereo frame
  * @return true if all ok
  * \param - output tf transform
  */
bool template_pose::TemplatePose::estimateTransform(tf::Transform& output)
{
  // Initialize output
  output.setIdentity();

  // Crosscheck descriptors matching
  Mat desc_image = img_prop_.getDesc();
  Mat desc_template = template_prop_.getDesc();
  vector<DMatch> matches;
  Mat match_mask;
  template_pose::OpencvUtils::crossCheckThresholdMatching(desc_image,
  desc_template, params_.desc_threshold, match_mask, matches);

  // Publish matches
  if (matches_image_pub_.getNumSubscribers() > 0)
  {
    Mat img_matches;
    drawMatches(img_prop_.getImg(), 
                img_prop_.getKp(), 
                template_prop_.getImg(), 
                template_prop_.getKp(), 
                matches, img_matches);

    cv_bridge::CvImage cv_image;
    cv_image.encoding = enc::BGR8;
    cv_image.image = img_matches;
    matches_image_pub_.publish(cv_image.toImageMsg());
  }

  ROS_INFO_STREAM("[TemplatePose:] Found " << matches.size() <<
   " matches (min_matches is: " << params_.min_matches << ")");

  // Sanity check
  if ((int)matches.size() < params_.min_matches)
    return false;

  // Extract keypoints and 3d points of vertex i and j
  vector<KeyPoint> kp_image = img_prop_.getKp();
  vector<Point3f> points3d_template = template_prop_.get3Dpoints();

  vector<Point2f> matched_keypoints;
  vector<Point3f> matched_3d_points;
  for (size_t i = 0; i < matches.size(); ++i)
  {
    int idx_image = matches[i].queryIdx;
    int idx_template = matches[i].trainIdx;
    matched_3d_points.push_back(points3d_template[idx_template]);
    matched_keypoints.push_back(kp_image[idx_image].pt);
  }
  
  // Compute the transformation between the vertices
  Mat rvec, tvec;
  vector<int> inliers;
  solvePnPRansac(matched_3d_points, matched_keypoints, camera_matrix_, 
                 Mat(), rvec, tvec, false /* no extrinsic guess */,
                 100 /* iterations */, 4.0 /* reproj. error */,
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

  output = tf::Transform(quaternion, translation);
  return true;
}

bool template_pose::TemplatePose::detectSrv(std_srvs::Empty::Request&, std_srvs::Empty::Response&)
{
  camera_to_template_.setIdentity();
  do_detection_ = false;
  toggle_detection_ = true;
  ROS_INFO("[TemplatePose:] Service Detect requested.");
  return true;
}

bool template_pose::TemplatePose::startDetectionSrv(std_srvs::Empty::Request&, std_srvs::Empty::Response&)
{
  camera_to_template_.setIdentity();
  do_detection_ = true;
  ROS_INFO("[TemplatePose:] Service Start Detection requested.");
  return true;
}

bool template_pose::TemplatePose::stopDetectionSrv(std_srvs::Empty::Request&, std_srvs::Empty::Response&)
{
  do_detection_ = false;
  ROS_INFO("[TemplatePose:] Service Stop Detection requested.");
  return true;
}

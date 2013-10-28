#include "image_properties.h"
#include "opencv_utils.h"
#include <ros/ros.h>

/** \brief Parameter constructor. Sets the parameter struct to default values.
  */
template_pose::ImageProperties::Params::Params() :
  desc_type("SIFT"),
  bucket_width(DEFAULT_BUCKET_WIDTH),
  bucket_height(DEFAULT_BUCKET_HEIGHT),
  max_bucket_features(DEFAULT_MAX_BUCKET_FEATURES),
  px_meter_x(DEFAULT_PX_METER_X),
  px_meter_y(DEFAULT_PX_METER_Y)
{}

/** \brief ImageProperties constructor
  */
template_pose::ImageProperties::ImageProperties() {}

/** \brief Sets the parameters
  * \param parameter struct.
  */
void template_pose::ImageProperties::setParams(const Params& params) 
{
  params_ = params;
}

/** \brief Return the image
  * @return image
  */
Mat template_pose::ImageProperties::getImg() { return img_; }

/** \brief Return the keypoints of the image
  * @return image keypoints
  */
vector<KeyPoint> template_pose::ImageProperties::getKp() { return kp_; }

/** \brief Return the descriptors of the image
  * @return image descriptors
  */
Mat template_pose::ImageProperties::getDesc() { return desc_; }

/** \brief Return the 3D points of the image
  * @return 3D points
  */
vector<Point3f> template_pose::ImageProperties::get3Dpoints() { return points_3d_; }

/** \brief Compute the properties for the images
  * \param image reference image.
  */
void template_pose::ImageProperties::setImg(const Mat& img)
{
  img_ = img;

  // Extract keypoints and descriptors of reference image
  desc_ = Mat_< vector<float> >();
  template_pose::OpencvUtils::keypointDetector(img_, kp_, params_.desc_type);

  // Bucket keypoints
  kp_ = template_pose::OpencvUtils::bucketKeypoints(kp_, 
                                                    params_.bucket_width, 
                                                    params_.bucket_height, 
                                                    params_.max_bucket_features);
  template_pose::OpencvUtils::descriptorExtraction(img_, kp_, desc_, params_.desc_type);
}

/** \brief Compute the 3D coordinates
  */
void template_pose::ImageProperties::compute3D()
{
  // Reset the 3D points
  points_3d_.clear();

  // Compute the 3D coordinates
  for (size_t i=0; i<kp_.size(); i++)
  {
    Point3f world_point;
    world_point.x = (float)kp_[i].pt.x/params_.px_meter_x;
    world_point.y = (float)kp_[i].pt.y/params_.px_meter_y;
    world_point.z = 0.0;
    points_3d_.push_back(world_point);
  }
}
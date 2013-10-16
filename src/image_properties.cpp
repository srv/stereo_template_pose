#include "image_properties.h"
#include "opencv_utils.h"
#include <ros/ros.h>

/** \brief Parameter constructor. Sets the parameter struct to default values.
  */
template_pose::ImageProperties::Params::Params() :
  desc_type("SIFT"),
  desc_threshold(DEFAULT_DESC_THRESHOLD),
  epipolar_threshold(DEFAULT_EPIPOLAR_THRESHOLD),
  bucket_width(DEFAULT_BUCKET_WIDTH),
  bucket_height(DEFAULT_BUCKET_HEIGHT),
  max_bucket_features(DEFAULT_MAX_BUCKET_FEATURES)  
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

/** \brief Return the left image
  * @return left image
  */
Mat template_pose::ImageProperties::getLeftImg() { return l_img_; }

/** \brief Return the right image
  * @return right image
  */
Mat template_pose::ImageProperties::getRightImg() { return r_img_; }

/** \brief Return the keypoints of the left image
  * @return image keypoints
  */
vector<KeyPoint> template_pose::ImageProperties::getLeftKp() { return l_kp_; }

/** \brief Return the keypoints of the right image
  * @return image keypoints
  */
vector<KeyPoint> template_pose::ImageProperties::getRightKp() { return r_kp_; }

/** \brief Return the keypoints of the left image after 3D matching
  * @return image keypoints
  */
vector<KeyPoint> template_pose::ImageProperties::getLeftKp3D() { return l_kp_3d_; }

/** \brief Return the keypoints of the right image after 3D matching
  * @return image keypoints
  */
vector<KeyPoint> template_pose::ImageProperties::getRightKp3D() { return r_kp_3d_; }

/** \brief Return the descriptors of the left image
  * @return image descriptors
  */
Mat template_pose::ImageProperties::getLeftDesc() { return l_desc_; }

/** \brief Return the descriptors of the right image
  * @return image descriptors
  */
Mat template_pose::ImageProperties::getRightDesc() { return r_desc_; }

/** \brief Return the descriptors of the left image after 3D matching
  * @return image descriptors
  */
Mat template_pose::ImageProperties::getLeftDesc3D() { return l_desc_3d_; }

/** \brief Return the descriptors of the right image after 3D matching
  * @return image descriptors
  */
Mat template_pose::ImageProperties::getRightDesc3D() { return r_desc_3d_; }

/** \brief Return the 3D points of the stereo pair
  * @return 3D points
  */
vector<Point3f> template_pose::ImageProperties::get3Dpoints() { return points_3d_; }

/** \brief Sets the stereo camera model for the class
  * \param stereo_camera_model.
  */
void template_pose::ImageProperties::setCameraModel(image_geometry::StereoCameraModel stereo_camera_model)
{
  stereo_camera_model_ = stereo_camera_model;
}

/** \brief Compute the properties for the images
  * \param image reference image.
  */
void template_pose::ImageProperties::setLeftImg(const Mat& img)
{
  l_img_ = img;

  // Extract keypoints and descriptors of reference image
  l_desc_ = Mat_< vector<float> >();
  template_pose::OpencvUtils::keypointDetector(l_img_, l_kp_, params_.desc_type);

  // Bucket keypoints
  l_kp_ = template_pose::OpencvUtils::bucketKeypoints(l_kp_, 
                                                    params_.bucket_width, 
                                                    params_.bucket_height, 
                                                    params_.max_bucket_features);

  template_pose::OpencvUtils::descriptorExtraction(l_img_, l_kp_, l_desc_, params_.desc_type);

  // Compute 3D if right image properties are set
  if (l_kp_.size() != 0 && r_kp_.size() != 0)
    compute3D();
}

/** \brief Compute the properties for the images
  * \param image reference image.
  */
void template_pose::ImageProperties::setRightImg(const Mat& img)
{
  r_img_ = img;

  // Extract keypoints and descriptors of reference image
  r_desc_ = Mat_< vector<float> >();
  template_pose::OpencvUtils::keypointDetector(r_img_, r_kp_, params_.desc_type);

  // Bucket keypoints
  r_kp_ = template_pose::OpencvUtils::bucketKeypoints(r_kp_, 
                                                    params_.bucket_width, 
                                                    params_.bucket_height, 
                                                    params_.max_bucket_features);

  template_pose::OpencvUtils::descriptorExtraction(r_img_, r_kp_, r_desc_, params_.desc_type);

  // Compute 3D if right image properties are set
  if (l_kp_.size() != 0 && r_kp_.size() != 0)
    compute3D();
}

/** \brief Computes the 3D points of the image
  */
void template_pose::ImageProperties::compute3D()
{

  // Find matches between left and right images
  vector<DMatch> matches, matches_filtered;
  Mat match_mask;
  template_pose::OpencvUtils::crossCheckThresholdMatching(l_desc_, 
      r_desc_, params_.desc_threshold, match_mask, matches);

  // Filter matches by epipolar 
  for (size_t i = 0; i < matches.size(); ++i)
  {
    if (abs(l_kp_[matches[i].queryIdx].pt.y - r_kp_[matches[i].trainIdx].pt.y) 
        < params_.epipolar_threshold)
      matches_filtered.push_back(matches[i]);
  }

  // Compute 3D points
  vector<KeyPoint> l_matched_kp, r_matched_kp;
  vector<Point3f> matched_3d_points;
  Mat l_matched_desc, r_matched_desc;
  for (size_t i = 0; i < matches_filtered.size(); ++i)
  {
    int index_left = matches_filtered[i].queryIdx;
    int index_right = matches_filtered[i].trainIdx;
    Point3d world_point;
    template_pose::OpencvUtils::calculate3DPoint( stereo_camera_model_,
                                                l_kp_[index_left].pt,
                                                r_kp_[index_right].pt,
                                                world_point);
    matched_3d_points.push_back(world_point);
    l_matched_kp.push_back(l_kp_[index_left]);
    r_matched_kp.push_back(r_kp_[index_right]);
    l_matched_desc.push_back(l_desc_.row(index_left));
    r_matched_desc.push_back(r_desc_.row(index_right));
  }

  // Save properties
  l_kp_3d_ = l_matched_kp;
  r_kp_3d_ = r_matched_kp;
  l_desc_3d_ = l_matched_desc;
  r_desc_3d_ = r_matched_desc;
  points_3d_ = matched_3d_points;
}
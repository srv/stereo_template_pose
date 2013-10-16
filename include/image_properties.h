#ifndef IMAGE_PROPERTIES_H
#define IMAGE_PROPERTIES_H

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <image_geometry/stereo_camera_model.h>

using namespace std;
using namespace cv;

namespace template_pose
{

class ImageProperties
{

public:

  // Class contructor
  ImageProperties();

  struct Params
  {
    //Default constructor sets all values to defaults.
    Params();

    // Class parameters
    string desc_type;               //!> Descriptor type can be: SIFT or SURF.
    double desc_threshold;          //!> Descriptor threshold (0-1).
    double epipolar_threshold;      //!> Maximum epipolar distance for stereo matching.
    int bucket_width;               //!> Bucket width.
    int bucket_height;              //!> Bucket height.
    int max_bucket_features;        //!> Maximum number the features per bucket.

    // Default values
    static const double         DEFAULT_DESC_THRESHOLD = 0.8;
    static const double         DEFAULT_EPIPOLAR_THRESHOLD = 3.5;
    static const int            DEFAULT_BUCKET_WIDTH = 30;
    static const int            DEFAULT_BUCKET_HEIGHT = 30;
    static const int            DEFAULT_MAX_BUCKET_FEATURES = 10;
  };

  // Set the parameter struct
  void setParams(const Params& params);

  // Return current parameters
  inline Params params() const { return params_; }

  // Set the stereo camera model
  void setCameraModel(image_geometry::StereoCameraModel stereo_camera_model);

  // Compute the keypoints and descriptors for the images
  void setLeftImg(const Mat& img);
  void setRightImg(const Mat& img);

  // Return the images
  Mat getLeftImg();
  Mat getRightImg();

  // Return the keypoints of the images
  vector<KeyPoint> getLeftKp();
  vector<KeyPoint> getRightKp();
  vector<KeyPoint> getLeftKp3D();
  vector<KeyPoint> getRightKp3D();

  // Return the descriptors of the images
  Mat getLeftDesc();
  Mat getRightDesc();
  Mat getLeftDesc3D();
  Mat getRightDesc3D();

  // Return the 3D points of the stereo pair
  vector<Point3f> get3Dpoints();

  // Computes the 3D points for the stereo pair
  void compute3D();

private:

  // Stores parameters
  Params params_;

  // Stereo vision properties
  Mat l_img_, r_img_;                   //!> Stores the images
  image_geometry::StereoCameraModel 
    stereo_camera_model_;               //!> Object to save the stereo camera model
  Mat camera_matrix_;                   //!> Used to save the camera matrix
  vector<KeyPoint> l_kp_, r_kp_;        //!> Unfiltered keypoints of the images.
  vector<KeyPoint> l_kp_3d_, r_kp_3d_;  //!> Keypoints of the images after 3D matching and triangulation.
  Mat l_desc_, r_desc_;                 //!> Unfiltered descriptors of the images.
  Mat l_desc_3d_, r_desc_3d_;           //!> Descriptors of the images after 3D matching and triangulation.
  vector<Point3f> points_3d_;           //!> 3D points of the stereo correspondences
};

} // namespace

#endif // IMAGE_PROPERTIES_H
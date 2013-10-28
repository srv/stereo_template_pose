#ifndef IMAGE_PROPERTIES_H
#define IMAGE_PROPERTIES_H

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

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
    string desc_type;                 //!> Descriptor type can be: SIFT or SURF.
    int bucket_width;                 //!> Bucket width.
    int bucket_height;                //!> Bucket height.
    int max_bucket_features;          //!> Maximum number the features per bucket.
    float px_meter_x;                 //!> Number of pixels per meters in x direction
    float px_meter_y;                 //!> Number of pixels per meters in y direction

    // Default values
    static const int            DEFAULT_BUCKET_WIDTH = 30;
    static const int            DEFAULT_BUCKET_HEIGHT = 30;
    static const int            DEFAULT_MAX_BUCKET_FEATURES = 10;
    static const float          DEFAULT_PX_METER_X = 2000;
    static const float          DEFAULT_PX_METER_Y = 2000;
  };

  // Set the parameter struct
  void setParams(const Params& params);

  // Return current parameters
  inline Params params() const { return params_; }

  // Compute the keypoints and descriptors for the images
  void setImg(const Mat& img);

  // Return the images
  Mat getImg();

  // Return the keypoints of the images
  vector<KeyPoint> getKp();

  // Return the descriptors of the images
  Mat getDesc();

  // Return the 3D points of the image
  vector<Point3f> get3Dpoints();

  // Computes the 3D points for the image
  void compute3D();

private:

  // Stores parameters
  Params params_;

  // Stereo vision properties
  Mat img_;                             //!> Stores the image
  vector<KeyPoint> kp_;                 //!> Unfiltered keypoints of the images.
  Mat desc_;                            //!> Unfiltered descriptors of the images.
  vector<Point3f> points_3d_;           //!> 3D points of the image correspondences
};

} // namespace

#endif // IMAGE_PROPERTIES_H
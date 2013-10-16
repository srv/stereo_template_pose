#ifndef OPENCV_UTILS
#define OPENCV_UTILS

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <std_msgs/Float32.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <image_geometry/stereo_camera_model.h>

using namespace std;
using namespace cv;

namespace template_pose
{

class OpencvUtils
{

public:

  /** \brief extract the keypoints of some image
    * @return 
    * \param image the source image
    * \param key_points is the pointer for the resulting image key_points
    * \param type descriptor type (see opencv docs)
    */
  static void keypointDetector( const Mat& image, 
                                vector<KeyPoint>& key_points, 
                                string type)
  {
    initModule_nonfree();
    Ptr<FeatureDetector> cv_detector;
    cv_detector = FeatureDetector::create(type);
    try
    {
      cv_detector->detect(image, key_points);
    }
    catch (Exception& e)
    {
      ROS_WARN("[StereoSlam:] cv_detector exception: %s", e.what());
    }
  }

  /** \brief extract descriptors of some image
    * @return 
    * \param image the source image
    * \param key_points keypoints of the source image
    * \param descriptors is the pointer for the resulting image descriptors
    */
  static void descriptorExtraction(const Mat& image,
   vector<KeyPoint>& key_points, Mat& descriptors, string type)
  {
    Ptr<DescriptorExtractor> cv_extractor;
    cv_extractor = DescriptorExtractor::create(type);
    try
    {
      cv_extractor->compute(image, key_points, descriptors);
    }
    catch (Exception& e)
    {
      ROS_WARN("[StereoSlam:] cv_extractor exception: %s", e.what());
    }
  }

  /** \brief match descriptors of 2 images by threshold
    * @return 
    * \param descriptors1 descriptors of image1
    * \param descriptors2 descriptors of image2
    * \param threshold to determine correct matchings
    * \param match_mask mask for matchings
    * \param matches output vector with the matches
    */
  static void thresholdMatching(const Mat& descriptors1, const Mat& descriptors2,
    double threshold, const Mat& match_mask, vector<DMatch>& matches)
  {
    matches.clear();
    if (descriptors1.empty() || descriptors2.empty())
      return;
    assert(descriptors1.type() == descriptors2.type());
    assert(descriptors1.cols == descriptors2.cols);

    const int knn = 2;
    Ptr<DescriptorMatcher> descriptor_matcher;
    // choose matcher based on feature type
    if (descriptors1.type() == CV_8U)
    {
      descriptor_matcher = DescriptorMatcher::create("BruteForce-Hamming");
    }
    else
    {
      descriptor_matcher = DescriptorMatcher::create("BruteForce");
    }
    vector<vector<DMatch> > knn_matches;
    descriptor_matcher->knnMatch(descriptors1, descriptors2,
            knn_matches, knn);

    for (size_t m = 0; m < knn_matches.size(); m++ )
    {
      if (knn_matches[m].size() < 2) continue;
      bool match_allowed = match_mask.empty() ? true : match_mask.at<unsigned char>(
          knn_matches[m][0].queryIdx, knn_matches[m][0].trainIdx) > 0;
      float dist1 = knn_matches[m][0].distance;
      float dist2 = knn_matches[m][1].distance;
      if (dist1 / dist2 < threshold && match_allowed)
      {
        matches.push_back(knn_matches[m][0]);
      }
    }
  }

  /** \brief filter matches of cross check matching
    * @return 
    * \param matches1to2 matches from image 1 to 2
    * \param matches2to1 matches from image 2 to 1
    * \param matches output vector with filtered matches
    */
  static void crossCheckFilter(
      const vector<DMatch>& matches1to2, 
      const vector<DMatch>& matches2to1,
      vector<DMatch>& checked_matches)
  {
    checked_matches.clear();
    for (size_t i = 0; i < matches1to2.size(); ++i)
    {
      bool match_found = false;
      const DMatch& forward_match = matches1to2[i];
      for (size_t j = 0; j < matches2to1.size() && match_found == false; ++j)
      {
        const DMatch& backward_match = matches2to1[j];
        if (forward_match.trainIdx == backward_match.queryIdx &&
            forward_match.queryIdx == backward_match.trainIdx)
        {
          checked_matches.push_back(forward_match);
          match_found = true;
        }
      }
    }
  }

  /** \brief match descriptors of 2 images by threshold
    * @return 
    * \param descriptors1 descriptors of image 1
    * \param descriptors2 descriptors of image 2
    * \param threshold to determine correct matchings
    * \param match_mask mask for matchings
    * \param matches output vector with the matches
    */
  static void crossCheckThresholdMatching(
    const Mat& descriptors1, const Mat& descriptors2,
    double threshold, const Mat& match_mask,
    vector<DMatch>& matches)
  {
    vector<DMatch> query_to_train_matches;
    thresholdMatching(descriptors1, descriptors2, threshold, match_mask, query_to_train_matches);
    vector<DMatch> train_to_query_matches;
    Mat match_mask_t;
    if (!match_mask.empty()) match_mask_t = match_mask.t();
    thresholdMatching(descriptors2, descriptors1, threshold, match_mask_t, train_to_query_matches);

    crossCheckFilter(query_to_train_matches, train_to_query_matches, matches);
  }

  /** \brief Compute the 3D point projecting the disparity
    * @return
    * \param stereo_camera_model is the camera model
    * \param left_point on the left image
    * \param right_point on the right image
    * \param world_point pointer to the corresponding 3d point
    */
  static void calculate3DPoint(const image_geometry::StereoCameraModel stereo_camera_model,
                               const Point2d& left_point, 
                               const Point2d& right_point, 
                               Point3d& world_point)
  {
    double disparity = left_point.x - right_point.x;
    stereo_camera_model.projectDisparityTo3d(left_point, disparity, world_point);
  }

  /** \brief Sort 2 descriptors matchings by distance
    * @return true if vector 1 is smaller than vector 2
    * \param descriptor matching 1
    * \param descriptor matching 2
    */
  static bool sortDescByDistance(const DMatch& d1, const DMatch& d2)
  {
    return (d1.distance < d2.distance);
  }

  /** \brief Sort 2 keypoints by response
    * @return true if vector 1 is smaller than vector 2
    * \param Keypoint 1
    * \param Keypoint 2
    */
  static bool sortKpByResponse(const KeyPoint& kp1, const KeyPoint& kp2)
  {
    return (kp1.response < kp2.response);
  }

  /** \brief Keypoints bucketing
    * @return vector of keypoints after bucketing filtering
    * \param kp vector of keypoints
    * \param b_width is the width of the buckets
    * \param b_height is the height of the buckets
    * \param b_num_feautres is the maximum number of features per bucket
    */
  static vector<KeyPoint> bucketKeypoints(vector<KeyPoint> kp, 
                                          int b_width, 
                                          int b_height, 
                                          int b_num_feautres)
  {
    // Find max values
    float x_max = 0;
    float y_max = 0;
    for (size_t i=0; i<kp.size(); i++)
    {
      if (kp[i].pt.x > x_max) x_max = kp[i].pt.x;
      if (kp[i].pt.y > y_max) y_max = kp[i].pt.y;
    }

    // Allocate number of buckets needed
    int bucket_cols = (int)floor(x_max/b_width) + 1;
    int bucket_rows = (int)floor(y_max/b_height) + 1;
    vector<KeyPoint> *buckets = new vector<KeyPoint>[bucket_cols*bucket_rows];

    // Assign keypoints to their buckets
    for (size_t i=0; i<kp.size(); i++)
    {
      int u = (int)floor(kp[i].pt.x/b_width);
      int v = (int)floor(kp[i].pt.y/b_height);
      buckets[v*bucket_cols+u].push_back(kp[i]);
    }

    // Refill keypoints from buckets
    vector<KeyPoint> output;
    for (int i=0; i<bucket_cols*bucket_rows; i++)
    {
      // Sort keypoints by response
      //sort(buckets[i].begin(), buckets[i].end(), template_pose::OpencvUtils::sortKpByResponse);

      // shuffle bucket indices randomly
      random_shuffle(buckets[i].begin(),buckets[i].end());
      
      // Add up to max_features features from this bucket to output
      int k=0;
      for (vector<KeyPoint>::iterator it=buckets[i].begin(); it!=buckets[i].end(); it++)
      {
        output.push_back(*it);
        k++;
        if (k >= b_num_feautres)
          break;
      }
    }
    return output;
  }

  /** \brief Matches bucketing
    * @return vector of matched keypoints after bucketing filtering
    * \param matches vector of matches
    * \param kp vector of keypoints
    * \param b_width is the width of the buckets
    * \param b_height is the height of the buckets
    * \param b_num_feautres is the maximum number of features per bucket
    */
  static vector<DMatch> bucketMatches(vector<DMatch> matches, 
                                      vector<KeyPoint> kp, 
                                      int b_width, 
                                      int b_height, 
                                      int b_num_feautres)
  {
    // Find max values
    float x_max = 0;
    float y_max = 0;
    for (vector<DMatch>::iterator it = matches.begin(); it!=matches.end(); it++)
    {
      if (kp[it->queryIdx].pt.x > x_max) x_max = kp[it->queryIdx].pt.x;
      if (kp[it->queryIdx].pt.y > y_max) y_max = kp[it->queryIdx].pt.y;
    }

    // Allocate number of buckets needed
    int bucket_cols = (int)floor(x_max/b_width) + 1;
    int bucket_rows = (int)floor(y_max/b_height) + 1;
    vector<DMatch> *buckets = new vector<DMatch>[bucket_cols*bucket_rows];

    // Assign matches to their buckets
    for (vector<DMatch>::iterator it = matches.begin(); it!=matches.end(); it++)
    {
      int u = (int)floor(kp[it->queryIdx].pt.x/b_width);
      int v = (int)floor(kp[it->queryIdx].pt.y/b_height);
      buckets[v*bucket_cols+u].push_back(*it);
    }

    // Refill matches from buckets
    vector<DMatch> output;
    for (int i=0; i<bucket_cols*bucket_rows; i++)
    {
      // Sort descriptors matched by distance
      sort(buckets[i].begin(), buckets[i].end(), template_pose::OpencvUtils::sortDescByDistance);
      
      // Add up to max_features features from this bucket to output
      int k=0;
      for (vector<DMatch>::iterator it=buckets[i].begin(); it!=buckets[i].end(); it++)
      {
        output.push_back(*it);
        k++;
        if (k >= b_num_feautres)
          break;
      }
    }
    return output;
  }

  /** \brief Show a tf::Transform in the command line
   * @return 
   * \param input is the tf::Transform to be shown
   */
  static void showTf(tf::Transform input)
  {
      tf::Vector3 tran = input.getOrigin();
    tf::Matrix3x3 rot = input.getBasis();
    tf::Vector3 r0 = rot.getRow(0);
    tf::Vector3 r1 = rot.getRow(1);
    tf::Vector3 r2 = rot.getRow(2);
    ROS_INFO_STREAM("[StereoSlam:]\n" << r0.x() << ", " << r0.y() << ", " << r0.z() << ", " << tran.x() <<
                    "\n" << r1.x() << ", " << r1.y() << ", " << r1.z() << ", " << tran.y() <<
                    "\n" << r2.x() << ", " << r2.y() << ", " << r2.z() << ", " << tran.z());
  }
};

} // namespace

#endif // OPENCV_UTILS



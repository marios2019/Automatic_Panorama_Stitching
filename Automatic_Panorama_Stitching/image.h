#pragma once
#include <vector>
#include <opencv2/core/core.hpp> // Mat
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/matchers.hpp"

using namespace cv;
using namespace cv::detail;

class Image
{
public:
	
	Image(Mat); // Constructor
	~Image(); // Destructor

	int getID() const; // Returns the image id

	void setImg(Mat); // Change img
	Mat getImg() const; // Returns the img

	Mat getImg_gray() const; // Returns the img_gray

	void setImageFeatures(vector <KeyPoint>, Mat); // Change ImageFeatures
	ImageFeatures getImageFeatures() const; // Returns ImageFeatures

	void setImg_Keypoint(Mat); // Change img_keypoint
	Mat getImg_Keypoint() const; // Returns the img_keypoint

	void setIntrinsics(CameraParams); // Change intrinsics
	CameraParams getIntrinsics() const; // Returns the intrinsics

	void print_img(Mat) const; // Print colour image in a window

private:

	void setImg_gray(Mat); // Change img_gray

	int id; // image id
	Mat img; // input image
	Mat img_gray; // input image in grayscale
	ImageFeatures features; // features produced by SIFT
	Mat img_keypoint; // image with keypoints drawn
	CameraParams intrinsics; // estimation of intrinsic calibration matrix

	static int idGen; // image ID generator
};
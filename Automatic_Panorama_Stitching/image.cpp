#include <opencv2/core/core.hpp> // Mat
#include <opencv2/imgproc/imgproc.hpp> // cvtColor
#include <opencv2/highgui/highgui.hpp> // imshow, namedWindow
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include <vector>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace cv::detail;

#include "image.h"

// Constructor
Image::Image(Mat imgval)
{
	id = idGen++;
	img = imgval.clone();
	cvtColor(imgval, img_gray, CV_BGR2GRAY);
	img_keypoint = NULL;
}

int Image::idGen = 0; // Initialize idGen

// Destructor
Image::~Image()
{
}

// Returns the image id
int Image::getID() const
{
	return id;
}

// Change img
void Image::setImg(Mat imgVal)
{
	img = imgVal.clone();
	setImg_gray(imgVal);
}

// Returns the img
Mat Image::getImg() const
{
	return img;
}

// Change img_gray
void Image::setImg_gray(Mat imgVal)
{
	cvtColor(imgVal, img_gray, CV_BGR2GRAY);
}

// Returns the img_gray
Mat Image::getImg_gray() const
{
	return img_gray;
}

void Image::setImageFeatures(vector <KeyPoint> keypointsVal, Mat descriptorsVal) // Change ImageFeatures
{
	features.keypoints = keypointsVal;
	features.descriptors = descriptorsVal.clone();
	features.img_idx = getID();
	features.img_size = getImg().size();
}

ImageFeatures Image::getImageFeatures() const // Returns ImageFeatures
{
	return features;
}
// Change img_keypoint
void Image::setImg_Keypoint(Mat img_keypointVal)
{
	img_keypoint = img_keypointVal.clone();
}

// Returns the img_keypoint
Mat Image::getImg_Keypoint() const
{
	return img_keypoint;
}

// Change intrinsics
void Image::setIntrinsics(CameraParams intrinsicsVal)
{
	intrinsics = intrinsicsVal;
}

// Returns the intrinsics
CameraParams Image::getIntrinsics() const
{
	return intrinsics;
}

// Print colour image in a window
void Image::print_img(Mat imgVal) const
{
	String str = "Image ";
	str.append(to_string(id));
	namedWindow(str, WINDOW_AUTOSIZE);
	imshow(str, imgVal);

	// Wait for a keystroke in the window
	waitKey(0); 
}
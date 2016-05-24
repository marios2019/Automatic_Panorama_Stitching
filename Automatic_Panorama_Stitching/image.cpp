#include <opencv2/core/core.hpp> // Mat
#include <opencv2/imgproc/imgproc.hpp> // cvtColor
#include <opencv2/highgui/highgui.hpp> // imshow, namedWindow
#include <vector>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

#include "image.h"

// Constructor
Image::Image(Mat imgval)
{
	id = idGen++;
	img = imgval.clone();
	cvtColor(imgval, img_gray, CV_BGR2GRAY);
	img_keypoint = NULL;
	descriptors = NULL;
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

// Change keypoints
void Image::setKeypoints(vector <KeyPoint> keypointsVal)
{
	keypoints = keypointsVal;
} 

// Returns the keypoints
vector <KeyPoint> Image::getKeypoints() const
{
	return keypoints;
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

// Change descriptor
void Image::setDescriptors(Mat descriptorVal)
{
	descriptors = descriptorVal.clone();
}

// Returns the descriptor
Mat Image::getDescriptors() const
{
	return descriptors;
}

// Change index
void Image::setIndex(vector <int> idxVal)
{
	if (index.size() < 6)
	{
		index.push_back(idxVal);
	}
	else
	{
		index.pop_back();
		index.push_back(idxVal);
		sort(index.begin(), index.end());
	}
}
vector <vector <int>> Image::getIndex() const // Returns the index
{
	return index;
}

vector <int> Image::getIndex(int posVal) const // Returns the one rwo of index
{
	vector <int> error;
	error.push_back(NULL);
	return ((0 <= posVal) && (posVal < 6) ? index[posVal] : error);
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
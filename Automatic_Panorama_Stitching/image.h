#pragma once
#include <vector>
#include <opencv2/core/core.hpp> // Mat
#include "opencv2/features2d/features2d.hpp" // KeyPoint

using namespace cv;

class Image
{
public:
	
	Image(Mat); // Constructor
	~Image(); // Destructor

	int getID() const; // Returns the image id

	void setImg(Mat); // Change img
	Mat getImg() const; // Returns the img

	Mat getImg_gray() const; // Returns the img_gray

	void setKeypoints(vector <KeyPoint>); // Change keypoints
	vector <KeyPoint> getKeypoints() const; // Returns the keypoints

	void setImg_Keypoint(Mat); // Change img_keypoint
	Mat getImg_Keypoint() const; // Returns the img_keypoint

	void setDescriptors(Mat); // Change descriptor
	Mat getDescriptors() const; // Returns the descriptor

	void setIndex(int); // Change index
	vector <int> getIndex() const; // Returns the index

	void print_img(Mat) const; // Print colour image in a window


private:

	void setImg_gray(Mat); // Change img_gray

	int id; // image id
	Mat img; // input image
	Mat img_gray; // input image in grayscale
	//pt: array of keypoint coordinates, size: keypoint diameter, angle: keypoint orientation, 
	//responce: keypoint strength, octave:	pyramid octave in which the keypoint has been detected
	vector <KeyPoint> keypoints; // keypoints extracted by SIFT algorithm
	Mat img_keypoint; // image with keypoints drawn
	Mat descriptors; // descriptor
	vector <int> index; //contains indexes to the 6 best matching images to the current image

	static int idGen; // image ID generator
};
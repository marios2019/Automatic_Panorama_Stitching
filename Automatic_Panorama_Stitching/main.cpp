#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

//Show Images
void show_image(vector <Mat>, int, String);
//Faeture Detection
void detect_features(vector<vector <KeyPoint>>&, vector <Mat>&, vector <Mat>);


int main(int argc, char** argv)
{
	if (argc < 2)
	{
		cout << "No input file!!!" << endl;
		return -1;
	}

	//Open input file
	ifstream file(argv[1]);
	
	//Extract image paths line by line
	string str;
	vector <Mat> input_img;
	while (getline(file, str))
	{
		input_img.push_back(imread(str, IMREAD_UNCHANGED));   // Read the file
	}
				
	if (input_img.empty())                              // check for invalid input
	{
		cout << "could not open or find the image" << std::endl;
		return -1;
	}
	show_image(input_img, CV_8UC3, "Input Images");

	//Feature Detection
	vector <vector <KeyPoint>> keypoints;
	vector <Mat> img_keypoints;
	detect_features(keypoints, img_keypoints, input_img);
	show_image(img_keypoints, CV_8UC3, "Keypoints");

	return 0;
}

//Show Images
void show_image(vector <Mat> tempImg, int Type, String str)
{
	//Resize to fit screen
	for (int i = 0; i < tempImg.size(); i++)
		resize(tempImg[i], tempImg[i], Size(402, 515), 0, 0, INTER_AREA);

	//Fit to one image, containing all the others
	int dstWidth = tempImg[0].cols * 4 + 40;
	int dstHeight = tempImg[0].rows * (((int) tempImg.size() / 4) + 1) + 20;

	Mat dst = Mat(dstHeight, dstWidth, Type, cv::Scalar(0, 0, 0));
	Rect roi = Rect(0, 0, tempImg[0].cols, tempImg[0].rows);
	Mat targetROI = dst(roi); 
	tempImg[0].copyTo(targetROI);
	for (int i = 1; i < tempImg.size(); i++)
	{
		if (i < 4)
		{
			targetROI = dst(Rect(i*tempImg[i].cols + i*10, 0, tempImg[i].cols, tempImg[i].rows));
			tempImg[i].copyTo(targetROI);
		}
		else
		{
			targetROI = dst(Rect((i-4)*tempImg[i].cols + (i-4)*10, tempImg[i].rows + 5, tempImg[i].cols, tempImg[i].rows));
			tempImg[i].copyTo(targetROI);
		}
	}
	
	namedWindow(str, WINDOW_AUTOSIZE);
	imshow(str, dst);

	waitKey(0); // Wait for a keystroke in the window
}

//Faeture Detection
void detect_features(vector<vector <KeyPoint>> &keypoints, vector <Mat> &img_keypoints, vector <Mat> input_img)
{
	//Detect the keypoints using SIFT Detector
	int minHessian = 400;

	SiftFeatureDetector detector(minHessian);

	//BGR to Gray
	vector <Mat> img;
	for (int i = 0; i < input_img.size(); i++)
	{
		Mat tmpimg;
		cvtColor(input_img[i], tmpimg, CV_BGR2GRAY);
		img.push_back(tmpimg);
	}

	for (int i = 0; i < img.size(); i++)
	{
		vector <KeyPoint> tmp;
		detector.detect(img[i], tmp);
		keypoints.push_back(tmp);
	}

	//Draw keypoints
	for (int i = 0; i < img.size(); i++)
	{
		Mat tmp;
		drawKeypoints(img[i], keypoints[i], tmp, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		img_keypoints.push_back(tmp);
	}

}
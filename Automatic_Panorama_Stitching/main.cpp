#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/opencv_modules.hpp"
#include <iostream>
#include <algorithm> 
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

// Show Images
void show_image(vector <Mat>, int, String);
// Feature Detection
void detect_features(vector<vector <KeyPoint>>&, vector <Mat>&, vector <Mat>);
// Feature Description and Matching
void describe_match_features(vector <KeyPoint>, vector <KeyPoint>, Mat, Mat, vector <DMatch>&, vector <DMatch>&, int, int);

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		cout << "No input file!!!" << endl;
		return -1;
	}

	// Open input file
	ifstream file(argv[1]);
	
	// Extract image paths line by line
	string str;
	vector <Mat> input_img;
	while (getline(file, str))
	{
		input_img.push_back(imread(str, IMREAD_UNCHANGED));
	}
				
	// Check for invalid input
	if (input_img.empty())                              
	{
		cout << "could not open or find the image" << std::endl;
		return -1;
	}
	show_image(input_img, CV_8UC3, "Input Images");

	// Feature Detection
	//pt: array of keypoint coordinates, size: keypoint diameter, angle: keypoint orientation, 
	//responce: keypoint strength, octave:	pyramid octave in which the keypoint has been detected
	vector <vector <KeyPoint>> keypoints;
	vector <Mat> img_keypoints;
	detect_features(keypoints, img_keypoints, input_img);
	show_image(img_keypoints, CV_8UC3, "Keypoints");

	// BGR to Gray
	vector <Mat> img;
	for (int i = 0; i < input_img.size(); i++)
	{
		Mat tmpimg;
		cvtColor(input_img[i], tmpimg, CV_BGR2GRAY);
		img.push_back(tmpimg);
	}

	// Feature description and FLANN matching
	vector <vector <DMatch>> matches, good_matches;
	for (int i = 0; i < (img.size() - 1); i++)
	{
		for (int j = i + 1; j < img.size(); j++)
		{
			vector <DMatch> tmp_matches, tmp_good_matches;
			describe_match_features(keypoints[i], keypoints[j], img[i], img[j], tmp_matches, tmp_good_matches, i, j);
			matches.push_back(tmp_matches);
			good_matches.push_back(tmp_good_matches);
		}
	}
    
	return 0;
}

// Show Images
void show_image(vector <Mat> tempImg, int Type, String str)
{
	// Resize to fit screen
	for (int i = 0; i < tempImg.size(); i++)
		resize(tempImg[i], tempImg[i], Size(402, 515), 0, 0, INTER_AREA);

	// Fit to one image, containing all the others
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

// Faeture Detection
void detect_features(vector<vector <KeyPoint>> &keypoints, vector <Mat> &img_keypoints, vector <Mat> input_img)
{
	// Detect the keypoints using SIFT Detector
	int minHessian = 400;

	SiftFeatureDetector detector(minHessian);

	// BGR to Gray
	vector <Mat> img;
	for (int i = 0; i < input_img.size(); i++)
	{
		Mat tmpimg;
		cvtColor(input_img[i], tmpimg, CV_BGR2GRAY);
		img.push_back(tmpimg);
	}

	// Feature Detection
	for (int i = 0; i < img.size(); i++)
	{
		vector <KeyPoint> tmp;
		detector.detect(img[i], tmp);
		keypoints.push_back(tmp);
	}

	// Draw keypoints
	for (int i = 0; i < img.size(); i++)
	{
		Mat tmp;
		drawKeypoints(img[i], keypoints[i], tmp, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		img_keypoints.push_back(tmp);
	}

}

// Feature Description and Matching
void describe_match_features(vector <KeyPoint> keypoints_1, vector <KeyPoint> keypoints_2, Mat img_1, Mat img_2, vector <DMatch> &matches, vector <DMatch> &good_matches, int x, int y)
{

	// Calculate descriptors (feature vectors)
	SiftDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;

	extractor.compute(img_1, keypoints_1, descriptors_1);
	extractor.compute(img_2, keypoints_2, descriptors_2);

	// Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	matcher.match(descriptors_1, descriptors_2, matches);

	// Calculation of minimum distance between keypoints
	vector <double> dist;
	for (int i = 0; i < matches.size(); i++)
	{
		dist.push_back(matches[i].distance);
	}
	sort(dist.begin(), dist.end());
	double min_dist = 100;
	if (min_dist > dist.front()) min_dist = dist.front();

	////Calculation of k = 4 nearest neighbour and minimum distance between keypoints
	//vector <double> k_nn;
	//double min_dist;
	//if (descriptors_1.rows <= 4)
	//{
	//	for (int i = 0; i < descriptors_1.rows; i++)
	//	{
	//		k_nn.push_back(matches[i].distance);
	//	}
	//	sort(k_nn.begin(), k_nn.end());
	//	min_dist = k_nn.back();
	//}
	//else
	//{
	//	for (int i = 0; i < 4; i++)
	//	{
	//		k_nn.push_back(matches[i].distance);
	//	}
	//	sort(k_nn.begin(), k_nn.end());

	//	for (int i = 4; i < descriptors_1.rows; i++)
	//	{
	//		if (matches[i].distance < k_nn.back())
	//		{
	//			k_nn.pop_back();
	//			k_nn.push_back(matches[i].distance);
	//			sort(k_nn.begin(), k_nn.end());
	//		}
	//	}

	//	min_dist = k_nn.back();
	//}

	// Draw only "good" matches
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance <= max(2*min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}

	// Draw only "good" matches
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	// Store img_matches
	String str = "images/SIFT_matches/";
	str.append("Images_");
	str.append(to_string(x+1));
	str.append("-");
	str.append(to_string(y+1));
	str.append("_Keypoints_matched_");
	str.append(to_string(good_matches.size()));
	str.append(".jpg");
	imwrite(str, img_matches);

}
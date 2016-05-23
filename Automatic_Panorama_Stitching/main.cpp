#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/opencv_modules.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

using namespace cv;
using namespace std;

#include "image.h" // Definition of class Image
#include "match.h" // Definition of class Match

// Show Images
void show_image(vector <Image>, int, String);
// Feature Detection
void detect_features(vector <Image>&);
// Feature Description and Matching
void describe_match_features(vector <Image>&, vector <Match>&);

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
	vector <Image> images;
	string str;
	while (getline(file, str))
	{
		Mat input_img;
		input_img = (imread(str, IMREAD_UNCHANGED));
		// Check for invalid input
		if (input_img.empty())
		{
			cout << "could not open or find the image" << endl;
			return -1;
		}
		
		Image tmp_img(input_img);
		images.push_back(tmp_img);
	}
				
	show_image(images, CV_8UC3, "Input Images");

	// Feature Detection
	detect_features(images);
	show_image(images, CV_8UC3, "Keypoints");

	// Feature description and FLANN matching
	vector <Match> matches;
	describe_match_features(images, matches);
	//vector <vector <DMatch>> matches, good_matches;
	//for (int i = 0; i < (images.size() - 1); i++)
	//{
	//	for (int j = i + 1; j < images.size(); j++)
	//	{
	//		vector <DMatch> tmp_matches, tmp_good_matches;
	//		describe_match_features(keypoints[i], keypoints[j], img[i], img[j], tmp_matches, tmp_good_matches, i, j);
	//		matches.push_back(tmp_matches);
	//		good_matches.push_back(tmp_good_matches);
	//	}
	//}
 //   
	system("PAUSE");

	return 0;
}

// Show Images
void show_image(vector <Image> img, int Type, String str)
{
	// Get images from vector
	vector <Mat> tmp_img;
	for (int i = 0; i < img.size(); i++)
	{
		if (str.compare("Input Images") == 0)
		{
			tmp_img.push_back(img[i].getImg());
		}
		else if (str.compare("Keypoints") == 0)
		{
			tmp_img.push_back(img[i].getImg_Keypoint());
		}
	}

	// Resize to fit screen
	for (int i = 0; i < tmp_img.size(); i++)
	{
		resize(tmp_img[i], tmp_img[i], Size(402, 515), 0, 0, INTER_AREA);
	}
		

	// Fit to one image, containing all the others
	int dstWidth = tmp_img[0].cols * 4 + 40;
	int dstHeight = tmp_img[0].rows * (((int)tmp_img.size() / 4) + 1) + 20;

	Mat dst = Mat(dstHeight, dstWidth, Type, cv::Scalar(0, 0, 0));
	Rect roi = Rect(0, 0, tmp_img[0].cols, tmp_img[0].rows);
	Mat targetROI = dst(roi); 
	tmp_img[0].copyTo(targetROI);
	for (int i = 1; i < tmp_img.size(); i++)
	{
		if (i < 4)
		{
			targetROI = dst(Rect(i*tmp_img[i].cols + i * 10, 0, tmp_img[i].cols, tmp_img[i].rows));
			tmp_img[i].copyTo(targetROI);
		}
		else
		{
			targetROI = dst(Rect((i - 4)*tmp_img[i].cols + (i - 4) * 10, tmp_img[i].rows + 5, tmp_img[i].cols, tmp_img[i].rows));
			tmp_img[i].copyTo(targetROI);
		}
	}
	
	namedWindow(str, WINDOW_AUTOSIZE);
	imshow(str, dst);

	waitKey(0); // Wait for a keystroke in the window

	// Store images
	str.insert(0, "images/");
	str.append(".jpg");
	imwrite(str, dst);
}

// Faeture Detection
void detect_features(vector <Image>& images)
{
	// Detect the keypoints using SIFT Detector
	int minHessian = 400;

	SiftFeatureDetector detector(minHessian);

	// Feature Detection
	for (int i = 0; i < images.size(); i++)
	{
		vector <KeyPoint> tmp;
		detector.detect(images[i].getImg_gray(), tmp);
		images[i].setKeypoints(tmp);
	}

	// Draw keypoints
	for (int i = 0; i < images.size(); i++)
	{
		Mat tmp;
		drawKeypoints(images[i].getImg_gray(), images[i].getKeypoints(), tmp, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		images[i].setImg_Keypoint(tmp);
	}

}

//Feature Description and Matching
void describe_match_features(vector <Image> &images, vector <Match> &matches)
{

	// Calculate descriptors (feature vectors)
	SiftDescriptorExtractor extractor;

	for (int i = 0; i < images.size(); i++)
	{
		Mat tmp_descriptors;
		extractor.compute(images[i].getImg_gray() , images[i].getKeypoints(), tmp_descriptors);
		images[i].setDescriptors(tmp_descriptors);
	}
	
	// Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	for (int i = 0; i < (images.size() - 1); i++)
	{
		for (int j = i + 1; j < images.size(); j++)
		{
			vector <DMatch> tmp_matches;
			matcher.match(images[i].getDescriptors(), images[j].getDescriptors(), tmp_matches);
			Match tmp(tmp_matches, images[i].getID(), images[j].getID());
			matches.push_back(tmp);
		}
	}

	// Find the best matches between two images
	for (int i = 0; i < matches.size(); i++)
	{
		vector <DMatch> tmp_matches;
		Mat tmp_descriptors;
		tmp_matches = matches[i].getMatches();
		int imgIdx1 = matches[i].getImgIdx1();
		int imgIdx2 = matches[i].getImgIdx2();
		tmp_descriptors = images[imgIdx1].getDescriptors().clone();

		// Calculation of minimum distance between keypoints
		double max_dist = 0; double min_dist = 100;
		for (int j = 0; j < tmp_descriptors.rows; j++)
		{
			double dist = tmp_matches[j].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		// Select only "good" matches
		vector< DMatch > tmp_good_matches;
		for (int j = 0; j < tmp_descriptors.rows; j++)
		{
			if (tmp_matches[j].distance <= max(2 * min_dist, 0.02))
			{
				tmp_good_matches.push_back(tmp_matches[j]);
			}
		}
		matches[i].setGood_Matches(tmp_good_matches);

		// Draw only "good" matches
		Mat tmp_img_matches;
		drawMatches(images[imgIdx1].getImg_gray(), images[imgIdx1].getKeypoints(), images[imgIdx2].getImg_gray(), images[imgIdx2].getKeypoints(),
			matches[i].getGood_Matches(), tmp_img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		matches[i].setImg_matches(tmp_img_matches);

		// Store img_matches
		String str = "images/SIFT_matches2/";
		str.append("Images_");
		str.append(to_string(imgIdx1));
		str.append("-");
		str.append(to_string(imgIdx2));
		str.append("_Keypoints_matched_");
		str.append(to_string(matches[i].getGood_Matches().size()));
		str.append(".jpg");
		imwrite(str, tmp_img_matches);
	}
}
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
#include <ctime> /* clock_t, clock, CLOCKS_PER_SEC */

using namespace cv;
using namespace std;

#include "image.h" // Definition of class Image
#include "match.h" // Definition of class Match

// Terminating condition for while loop
bool isValidState(int);
// Show Images
void show_image(vector <Image>, int, String);
// Global Histogram Equalization
Mat hist_equalize(Mat);
// Feature Detection
void detect_features(vector <Image>&, bool);
// Feature Description and Matching
void describe_match_features(vector <Image>&, vector <Match>&, bool);

int main(int argc, char** argv)
{
	// Program initialization
	int option = -1;
	string input;
	bool flag = 0;
	vector <Image> images;
	vector <Match> matches;

	while (isValidState(option))
	{
		cout << "1. Provide input images." << endl;
		cout << "2. Apply global histogram equalazition in each image (Optional)." << endl;
		cout << "3. Apply SIFT algorithm to detect feautures." << endl;
		cout << "4. Feature Matching." << endl;
		cout << "0. Exit." << endl;
		cin >> option;
		cout << endl;

		switch (option)
		{
			// Program exit
			case 0:
			{
					cout << "Program endend by user!!!" << endl;
					break;
			}
			// Get images from input file
			case 1:
			{
					  cout << "Provide input images file with .txt extension: ";
					  cin >> input;

					  // Open input file
					  ifstream file(input);
					  // Check if input is empty or the file is empty
					  while ((!file.is_open()) || (file.peek() == ifstream::traits_type::eof()))
					  {
						  cout << "Provide input images file with .txt extension and " << endl;
						  cout << "make sure it's not empty: ";
						  cin >> input;
						  file.open(input);
					  }					 

					  // Extract image paths line by line
					  string str;
					  while (getline(file, str))
					  {
						  Mat input_img;
						  input_img = (imread(str, IMREAD_UNCHANGED));
						  // Check for invalid input
						  if (input_img.empty())
						  {
							  cout << "Could not open or find the image!!!" << endl;
							  cout << "Program terminated!!!" << endl;
							  return (0);
						  }

						  Image tmp_img(input_img);
						  images.push_back(tmp_img);
					  }

					  //Original images
					  flag = 0;
					  show_image(images, CV_8UC3, "Input Images");
					  break;
			}
			// Global Histogram Equalization
			case 2:
			{
					  // Check if input images have been provided
					  if (images.empty())
					  {
						  cout << "Please provide input images." << endl;
						  break;
					  }

					  // Images have been processed
					  flag = 1;
					  for (int i = 0; i < images.size(); i++)
						  images[i].setImg(hist_equalize(images[i].getImg()));
					  
					  show_image(images, CV_8UC3, "Input Images with Global Histogram Equalization");
					  break;
			}
			// Feature Detection
			case 3:
			{
					  // Check if input images have been provided
					  if (images.empty())
					  {
						  cout << "Please provide input images." << endl;
						  break;
					  }

					  // Timing Feature Detection
					  clock_t timerFD;
					  timerFD = clock();
					  detect_features(images, flag);
					  show_image(images, CV_8UC3, "Keypoints");
					  timerFD = clock() - timerFD;
					  cout << "Feature Detection time: " << ((float)timerFD / CLOCKS_PER_SEC) << " seconds." << endl;
					  break;
			}
			// Feature description and FLANN matching
			case 4:
			{
					  // Check if input images have been provided
					  if (images.empty())
					  {
						  cout << "Please provide input images." << endl;
						  break;
					  }

					  // Check if all images features have been extracted
					  for (int i = 0; i < images.size(); i++)
					  {
						  if (images[i].getKeypoints().empty())
						  {
							  cout << "Features from all images have to be extracted before feature matching!!!" << endl;
							  break;
						  }
					  }

					  // Timing Feature description and FLANN matching
					  clock_t timerM;
					  timerM = clock();
					  describe_match_features(images, matches, flag);
					  timerM = clock() - timerM;
					  cout << "Feature description and FLANN matching time: " << ((float)timerM / CLOCKS_PER_SEC) << " seconds." << endl;
					  break;
			}

			default: 
				continue;
		}
		
		cout << endl;
	}

	system("PAUSE");
	return(0);
}

// Terminating condition for while loop
bool isValidState(int option)
{
	return option != 0;
}

// Show Images
void show_image(vector <Image> img, int Type, String str)
{
	// Get images from vector
	vector <Mat> tmp_img;
	for (int i = 0; i < img.size(); i++)
	{
		if ((str.compare("Input Images") == 0) || (str.compare("Input Images with Global Histogram Equalization") == 0))
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
	
	// Show images
	imshow(str, dst);
	waitKey(1);
	cout << str << " images can been in a new window." << endl;

	// Store images
	str.insert(0, "images/");
	str.append(".jpg");
	imwrite(str, dst);
}

// Global Histogram Equalization
Mat hist_equalize(Mat input_image)
{
	if (input_image.channels() >= 3)
	{
		Mat ycrcb;

		cvtColor(input_image, ycrcb, CV_BGR2YCrCb);

		vector<Mat> channels;
		split(ycrcb, channels);

		equalizeHist(channels[0], channels[0]);

		Mat result;
		merge(channels, ycrcb);

		cvtColor(ycrcb, result, CV_YCrCb2BGR);

		return result;
	}
		
	cout << "The image has to be consist by at least 3 channels." << endl;
	return input_image;
}

// Faeture Detection
void detect_features(vector <Image>& images, bool flag)
{
	// Detect the keypoints using SIFT Detector
	int minHessian = 500;

	SiftFeatureDetector detector(minHessian);

	for (int i = 0; i < images.size(); i++)
	{
		// Feature Detection
		vector <KeyPoint> tmp_keypoints;
		detector.detect(images[i].getImg_gray(), tmp_keypoints);
		images[i].setKeypoints(tmp_keypoints);

		// Draw keypoints
		Mat tmp_img_keypoints;
		drawKeypoints(images[i].getImg_gray(), images[i].getKeypoints(), tmp_img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		images[i].setImg_Keypoint(tmp_img_keypoints);

		// Store img_keypoints
		string str;
		if (flag == 0)
		{
			str = "images/SIFT_Keypoints/Original_Image/";			
		}
		else
		{
			str = "images/SIFT_Keypoints/Histogram_Equalazition/";
		}
		str.append("Image_");
		str.append(to_string(images[i].getID()));
		str.append("_Keypoints_detected_");
		str.append(to_string(images[i].getKeypoints().size()));
		str.append(".jpg");
		imwrite(str, tmp_img_keypoints);
	}
}

//Feature Description and Matching
void describe_match_features(vector <Image> &images, vector <Match> &matches, bool flag)
{

	// Calculate descriptors (feature vectors)
	SiftDescriptorExtractor extractor;

	for (int i = 0; i < images.size(); i++)
	{
		Mat tmp_descriptors;
		extractor.compute(images[i].getImg_gray() , images[i].getKeypoints(), tmp_descriptors);
		images[i].setDescriptors(tmp_descriptors);
	}
	
	
	FlannBasedMatcher matcher;
	for (int i = 0; i < (images.size() - 1); i++)
	{
		for (int j = i + 1; j < images.size(); j++)
		{
			vector <DMatch> tmp_matches;
			Mat tmp_descriptors;
			int imgIdx1 = images[i].getID();
			int imgIdx2 = images[j].getID();
			tmp_descriptors = images[imgIdx1].getDescriptors().clone();

			// Matching descriptor vectors using FLANN matcher
			matcher.match(images[imgIdx1].getDescriptors(), images[imgIdx2].getDescriptors(), tmp_matches);
			Match tmp_match(tmp_matches, imgIdx1, imgIdx2);

			// Calculation of minimum distance between keypoints
			double max_dist = 0; double min_dist = 100;
			for (int k = 0; k < tmp_descriptors.rows; k++)
			{
				double dist = tmp_matches[k].distance;
				if (dist < min_dist)
				{
					min_dist = dist;
				}				
				if (dist > max_dist)
				{
					max_dist = dist;
				}
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
			tmp_match.setGood_Matches(tmp_good_matches);

			// Draw only "good" matches
			Mat tmp_img_matches;
			drawMatches(images[imgIdx1].getImg_gray(), images[imgIdx1].getKeypoints(), images[imgIdx2].getImg_gray(), images[imgIdx2].getKeypoints(),
				tmp_match.getGood_Matches(), tmp_img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			tmp_match.setImg_matches(tmp_img_matches);

			// Store img_matches
			string str;
			if (flag == 0)
			{
				str = "images/SIFT_Matches/Original_Image/";
			}
			else
			{
				str = "images/SIFT_Matches/Histogram_Equalazition/";
			}
			str.append("Images_");
			str.append(to_string(imgIdx1));
			str.append("-");
			str.append(to_string(imgIdx2));
			str.append("_Keypoints_matched_");
			str.append(to_string(tmp_match.getGood_Matches().size()));
			str.append(".jpg");
			imwrite(str, tmp_img_matches);

			// Store tmp_match to vector matches
			matches.push_back(tmp_match);
		}
	}
}
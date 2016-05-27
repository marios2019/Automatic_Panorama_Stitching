#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <ctime> /* clock_t, clock, CLOCKS_PER_SEC */

using namespace cv;
using namespace cv::detail;
using namespace std;

#include "image.h" // Definition of class Image

bool try_gpu = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
string ba_cost_func = "reproj";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_type = WAVE_CORRECT_HORIZ;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;
string result_name = "result.jpg";

// Global Histogram Equalization
Mat hist_equalize(Mat);
// Feature Detection
void det_desc_features(vector <Image>&, bool);
// Feature matching
void match_features(vector <MatchesInfo>&, vector <Image>, bool, float);
// Reject noise images which match to no other images
bool imageValidate(vector <MatchesInfo>, vector <Image>&, vector <Mat>&, float);
// Estimate homography and bundle adjustemnt
vector <double> homogr_ba(vector <Image>&, vector <MatchesInfo>, float);
// Wave correction
void wave_correct(vector <Image>&);

int main(int argc, char** argv)
{
	// Program initialization
	clock_t timerOverall;
	timerOverall = clock();
	bool hist = 0;
	string input;
	vector <Image> images;
	vector <Mat> images_scale;
	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
	double seam_work_aspect = 1;
	double compose_work_aspect = 1;
	string result_name = "result.jpg";
	
	if (argc < 2)
	{
		cout << "No input images file provided." << endl;
		system("PAUSE");
		return (0);
	}

	// Open input file
	ifstream file(argv[1]);
	// Check if input is empty or the file is empty
	if ((!file.is_open()) || (file.peek() == ifstream::traits_type::eof()))
	{
		cout << "Provide input images file with .txt extension and " << endl;
		cout << "make sure it's not empty: ";
		system("PAUSE");
		return (0);
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

		if (work_megapix < 0)
		{
			work_scale = 1;
			is_work_scale_set = true;
		}
		else
		{
			Mat tmp;
			if (!is_work_scale_set)
			{
				work_scale = min(1.0, sqrt(work_megapix * 1e6 / tmp_img.getImg().size().area()));
				is_work_scale_set = true;
			}
			resize(tmp_img.getImg(), tmp, Size(), work_scale, work_scale);
		}
		if (!is_seam_scale_set)
		{
			seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / tmp_img.getImg().size().area()));
			seam_work_aspect = seam_scale / work_scale;
			is_seam_scale_set = true;
		}
		
		Mat tmp;
		resize(tmp_img.getImg(), tmp, Size(), seam_scale, seam_scale);
		images_scale.push_back(tmp);
	}

	// Global Histogram Equalization
	// Images have been processed
	if (hist == 1)
	{
		for (int i = 0; i < images.size(); i++)
			images[i].setImg(hist_equalize(images[i].getImg()));
	}
	
	// Feature Detection and Descirption
	// Timing 
	clock_t timerFDD;
	timerFDD = clock();	
	cout << "Feature Detection and Descirption..." << endl;
	det_desc_features(images, hist);
	//show_image(images, CV_8UC3, "Keypoints");
	timerFDD = clock() - timerFDD;
	cout << "Feature Detection and Description time: " << ((float)timerFDD / CLOCKS_PER_SEC) << " seconds." << endl;
	cout << endl;

	// Feature Matching
	// Timing
	clock_t timerFM;
	timerFM = clock();
	vector <MatchesInfo> pairwise_matches;
	cout << "Pairwise image matching..." << endl;
	float match_conf = 0.65f;
	match_features(pairwise_matches, images, hist, match_conf);
	timerFM = clock() - timerFM;
	cout << "Pairwise image matching time: " << ((float)timerFM / CLOCKS_PER_SEC) << " seconds." << endl;
	cout << endl;

	// Reject noise images which match to no other images
	cout << "Reject noise images which match to no other images..." << endl;
	float conf_thresh = 1.f;
	bool flag = imageValidate(pairwise_matches, images, images_scale, conf_thresh);
	if (flag)
	{
		system("PAUSE");
		return (0);
	}
	cout << "Images that belong to panorama are: ";
	for (size_t i = 0; i < images.size(); i++)
	{
		cout << " #" << images[i].getID();
	}
	cout << endl;

	// Estimate Homography and bundle adjustement
	clock_t timerHBA;
	timerHBA = clock();
	cout << "Estimating Homography and bundle adjust camera intrinsics..." << endl;
	vector <double> focals;
	focals = homogr_ba(images, pairwise_matches, conf_thresh);
	timerHBA = clock() - timerHBA;
	cout << "Estimating Homography and bundle adjust camera intrinsics time: " << ((float)timerHBA / CLOCKS_PER_SEC) << " seconds." << endl;
	cout << endl;
	
	// Find median focal length
	sort(focals.begin(), focals.end());
	float warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	// Wave correction
	wave_correct(images);
////////

	cout << "Warping images (auxiliary)... " << endl;

	vector<Point> corners(images.size());
	vector<Mat> masks_warped(images.size());
	vector<Mat> images_warped(images.size());
	vector<Size> sizes(images.size());
	vector<Mat> masks(images.size());

	// Preapre images masks
	for (int i = 0; i < images.size(); ++i)
	{
		masks[i].create(images_scale[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	// Warp images and their masks

	Ptr<WarperCreator> warper_creator;
	string warp_type = "spherical";
	
	if (warp_type == "plane") warper_creator = new cv::PlaneWarper();
	else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarper();
	else if (warp_type == "spherical") warper_creator = new cv::SphericalWarper();

	if (warper_creator.empty())
	{
		cout << "Can't create the following warper '" << warp_type << "'\n";
		return 1;
	}

	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

	for (int i = 0; i < images.size(); ++i)
	{
		Mat_<float> K;
		images[i].getIntrinsics().K().convertTo(K, CV_32F);
		float swa = (float)seam_work_aspect;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;

		corners[i] = warper->warp(images_scale[i], K, images[i].getIntrinsics().R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, images[i].getIntrinsics().R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}

	vector<Mat> images_warped_f(images.size());
	for (int i = 0; i < images.size(); ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);

	Ptr<SeamFinder> seam_finder;
	if (seam_find_type == "no")
		seam_finder = new detail::NoSeamFinder();
	else if (seam_find_type == "voronoi")
		seam_finder = new detail::VoronoiSeamFinder();
	else if (seam_find_type == "gc_color")
	{
			seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
	}
	else if (seam_find_type == "gc_colorgrad")
	{
			seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
	}
	else if (seam_find_type == "dp_color")
		seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
	else if (seam_find_type == "dp_colorgrad")
		seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);
	if (seam_finder.empty())
	{
		cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
		return 1;
	}

	seam_finder->find(images_warped_f, corners, masks_warped);

	// Release unused memory
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();

	cout << "Compositing..." << endl;

	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask, mask, mask_warped;
	Ptr<Blender> blender;

	for (int  i = 0; i < images.size(); i++)
	{
		cout << "Compositing image #" << images[i].getID() << endl;

		// Read image and resize it if necessary
		if (!is_compose_scale_set)
		{
			is_compose_scale_set = true;

			// Compute relative scales
			//compose_seam_aspect = compose_scale / seam_scale;
			compose_work_aspect = compose_scale / work_scale;

			// Update warped image scale
			warped_image_scale *= static_cast<float>(compose_work_aspect);
			warper = warper_creator->create(warped_image_scale);

			// Update corners and sizes
			for (int i = 0; i < images.size(); ++i)
			{
				// Update intrinsics
				CameraParams camera = images[i].getIntrinsics();
				camera.focal *= compose_work_aspect;
				camera.ppx *= compose_work_aspect;
				camera.ppy *= compose_work_aspect;

				// Update corner and size
				Size sz = images[i].getImg().size();
				if (std::abs(compose_scale - 1) > 1e-1)
				{
					sz.width = cvRound(images[i].getImg().rows * compose_scale);
					sz.height = cvRound(images[i].getImg().cols * compose_scale);
				}

				Mat K;
				camera.K().convertTo(K, CV_32F);
				Rect roi = warper->warpRoi(sz, K, camera.R);
				corners[i] = roi.tl();
				sizes[i] = roi.size();
			}
		}
		//if (abs(compose_scale - 1) > 1e-1)
		//	resize(full_img, img, Size(), compose_scale, compose_scale);
		//else
		//	img = full_img;
		//full_img.release();
		//Size img_size = img.size();

		Mat K;
		images[i].getIntrinsics().K().convertTo(K, CV_32F);

		// Warp the current image
		warper->warp(images[i].getImg(), K, images[i].getIntrinsics().R, INTER_LINEAR, BORDER_REFLECT, img_warped);

		// Warp the current image mask
		mask.create(images[i].getImg().size(), CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, images[i].getIntrinsics().R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

		// Compensate exposure
		compensator->apply(i, corners[i], img_warped, mask_warped);

		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		mask.release();

		dilate(masks_warped[i], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size());
		mask_warped = seam_mask & mask_warped;

		if (blender.empty())
		{
			blender = Blender::createDefault(blend_type, false);
			Size dst_sz = resultRoi(corners, sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
			if (blend_width < 1.f)
				blender = Blender::createDefault(Blender::NO, false);
			else if (blend_type == Blender::MULTI_BAND)
			{
				MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
				mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
				cout << "Multi-band blender, number of bands: " << mb->numBands() << endl;
			}
			else if (blend_type == Blender::FEATHER)
			{
				FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
				fb->setSharpness(1.f / blend_width);
				cout << "Feather blender, sharpness: " << fb->sharpness() << endl;
			}
			blender->prepare(corners, sizes);
		}

		// Blend the current image
		blender->feed(img_warped_s, mask_warped, corners[i]);
	}

	Mat result, result_mask;
	blender->blend(result, result_mask);

	imwrite(result_name, result);

	timerOverall = clock() - timerOverall;
	cout << "Overall time: " << ((float)timerOverall / CLOCKS_PER_SEC) << " seconds." << endl;

	system("PAUSE");
	return(0);
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

// Faeture Detection and Decription
void det_desc_features(vector <Image>& images, bool flag)
{
	// Detect the keypoints using SIFT Detector
	SurfFeatureDetector detector;
	// Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	for (size_t i = 0; i < images.size(); i++)
	{
		// Feature Detection
		vector <KeyPoint> tmp_keypoints;
		detector.detect(images[i].getImg_gray(), tmp_keypoints);

		cout << "Features detected in image #" << i << " : " << tmp_keypoints.size() << endl;
		// Feature Description
		Mat tmp_descriptors;
		extractor.compute(images[i].getImg_gray(), tmp_keypoints, tmp_descriptors);

		// Store keypoints and descriptors
		images[i].setImageFeatures(tmp_keypoints, tmp_descriptors);

		// Draw keypoints
		Mat tmp_img_keypoints;
		drawKeypoints(images[i].getImg_gray(), tmp_keypoints, tmp_img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
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
		str.append(to_string(tmp_keypoints.size()));
		str.append(".jpg");
		imwrite(str, tmp_img_keypoints);
	}
}

// Feature Matching
void match_features(vector <MatchesInfo> &pairwise_matches, vector <Image> images, bool flag, float match_conf)
{
	// Get ImageFeatures
	vector <ImageFeatures> features;
	for (int i = 0; i < images.size(); i++)
	{
		features.push_back(images[i].getImageFeatures());
	}
	
	// Timing 
	clock_t timerMatch;
	timerMatch = clock();
	BestOf2NearestMatcher matcher(false, match_conf);
	matcher(features, pairwise_matches);
	timerMatch = clock() - timerMatch;
	cout << "Pairwise Matching: " << ((float)timerMatch / CLOCKS_PER_SEC) << " seconds." << endl;
	matcher.collectGarbage();

	for (size_t i = 0; i < pairwise_matches.size(); i++)
	{
		int src_img_idx = pairwise_matches[i].src_img_idx;
		int dst_img_idx = pairwise_matches[i].dst_img_idx;

		if ((src_img_idx != dst_img_idx) && (src_img_idx <= dst_img_idx))
		{
			// Draw only "good" matches
			Mat tmp_img_matches;
			drawMatches(images[src_img_idx].getImg_gray(), features[src_img_idx].keypoints, images[dst_img_idx].getImg_gray(), features[dst_img_idx].keypoints,
				pairwise_matches[i].matches, tmp_img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

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
			str.append(to_string(src_img_idx));
			str.append("-");
			str.append(to_string(dst_img_idx));
			str.append("_Keypoints_matched_");
			str.append(to_string(pairwise_matches[i].matches.size()));
			str.append(".jpg");
			imwrite(str, tmp_img_matches);
		}
	}
}

// Reject noise images which match to no other images
bool imageValidate(vector <MatchesInfo> pairwise_matches, vector <Image> &images, vector <Mat> &images_scale, float conf_thresh)
{
	vector <ImageFeatures> features;
	for (int i = 0; i < images.size(); i++)
	{
		features.push_back(images[i].getImageFeatures());
	}
		
	// Leave only images we are sure are from the same panorama
	vector <int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	vector <Image> tmp_images;
	vector <Mat> tmp_img_sc;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		tmp_images.push_back(images[indices[i]]);
		tmp_img_sc.push_back(images_scale[indices[i]]);
	}

	images = tmp_images;
	images_scale = tmp_img_sc;

	// Check if we still have enough images
	if (images.size() < 2)
	{
		cout << "Need more images" << endl;
		return 1;
	}
	else
	{
		return 0;
	}
}

// Estimate homography and bundle adjustemnt
vector <double> homogr_ba(vector <Image> &images, vector <MatchesInfo> pairwise_matches, float conf_thresh)
{
	vector <ImageFeatures> features;
	for (size_t i = 0; i < images.size(); i++)
	{
		features.push_back(images[i].getImageFeatures());
	}

	HomographyBasedEstimator estimator;
	vector <CameraParams> cameras;
	estimator(features, pairwise_matches, cameras);

	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		cout << "Initial intrinsics #" << images[i].getID() << ":\n" << cameras[i].K() << endl;
	}

	// Bundle adjustement
    BundleAdjusterReproj adjuster;
	adjuster.setConfThresh(conf_thresh);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
	adjuster.setRefinementMask(refine_mask);
	adjuster(features, pairwise_matches, cameras);

	vector <double> focals;
	for (size_t i = 0; i < images.size(); ++i)
	{
		cout << "Camera #" << images[i].getID() << ":\n" << cameras[i].K() << endl;
		focals.push_back(cameras[i].focal);
		images[i].setIntrinsics(cameras[i]);
	}

	return focals;
}

// Wave correction
void wave_correct(vector <Image> &images)
{
	vector <Mat> rmats;
	for (size_t i = 0; i < images.size(); ++i)
		rmats.push_back(images[i].getIntrinsics().R.clone());
	waveCorrect(rmats, wave_type);
	for (size_t i = 0; i < images.size(); ++i)
		images[i].getIntrinsics().R = rmats[i];
}
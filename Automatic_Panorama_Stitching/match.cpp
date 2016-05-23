#include <vector>
#include <opencv2/core/core.hpp> 
#include "opencv2/features2d/features2d.hpp" // DMatch

using namespace cv;

#include "match.h"

Match::Match(vector <DMatch> matchesVal, int imgIdx1Val, int imgIdx2Val)
{
	matches = matchesVal;
	imgIdx1 = imgIdx1Val;
	imgIdx2 = imgIdx2Val;
}

// Destructor
Match::~Match()
{
}

// Change matches
void Match::setMatches(vector <DMatch> matchesVal)
{
	matches = matchesVal;
}

// Returns the matches
vector <DMatch> Match::getMatches() const
{
	return matches;
}

// Change matches
void Match::setGood_Matches(vector <DMatch> good_matchesVal)
{
	good_matches = good_matchesVal;
}

// Returns the good_matches
vector <DMatch> Match::getGood_Matches() const
{
	return good_matches;
}

// Change imgIdx1
void Match::setImgIdx1(int imgIdx1Val)
{
	imgIdx1 = imgIdx1Val;
}

// Returns the imgIdx1
int Match::getImgIdx1() const
{
	return imgIdx1;
}

// Change imgIdx2
void Match::setImgIdx2(int imgIdx2Val)
{
	imgIdx2 = imgIdx2Val;
}

// Returns the imgIdx2
int Match::getImgIdx2() const
{
	return imgIdx2;
}

// Change img_matches
void Match::setImg_matches(Mat img_matchesVal)
{
	img_matches = img_matchesVal.clone();
}

// Returns the img_matches
Mat Match::getImg_matches() const
{
	return img_matches;
}
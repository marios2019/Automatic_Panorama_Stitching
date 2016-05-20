#pragma once
#include <vector>
#include "opencv2/features2d/features2d.hpp" // DMatch

using namespace cv;

class Match
{
public:

	Match(vector <DMatch>, int, int); // Constructor
	~Match(); //Destructor

	void setMatches(vector <DMatch>); // Change matches
	vector <DMatch> getMatches() const; // Returns the matches

	void setGood_Matches(vector <DMatch>); // Change good_matches
	vector <DMatch> getGood_Matches() const; // Returns the good_matches

	void setImgIdx1(int); // Change imgIdx1
	int getImgIdx1() const; // Returns the imgIdx1

	void setImgIdx2(int); // Change imgIdx2
	int getImgIdx2() const; // Returns the imgIdx

private:

	vector <DMatch> matches; // Vector which contains image keypoint matches
	vector <DMatch> good_matches; // Vector which contains image keypoint matches within a desirable distance
	int imgIdx1; // index pointing to query image
	int	imgIdx2; // index pointing to train image
};
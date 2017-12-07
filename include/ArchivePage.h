#pragma once

#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"

// ConstantS for neural network setting
#define INPUT_NEURONS 784
#define HIDDEN_NEURONS 20
#define OUTPUT_NEURONS 10
#define TRAINING_EPOCHS 2000 
#define ACCURACY 90

// Constants for digit detection image setting
#define DIGIT_NORMALISED_WIDTH 32
#define DIGIT_NORMALISED_HEIGHT 48
#define DIGIT_THRESHOLD_MIN 140
#define DIGIT_THRESHOLD_MAX 255 
#define DIGIT_SAMPLE_HEIGHT 20
#define DIGIT_SAMPLE_WIDTH 20

class ArchivePage
{

public:

	ArchivePage();
	~ArchivePage();

	void savePage(std::string filename);
	void loadPage(std::string filename);


public:
	
	int archiveDate;

	// lets try to load too
	std::vector< std::pair< cv::Point, cv::Point> > redRuleV;
	std::vector< std::pair< cv::Point, cv::Point> > blueRuleH;
	std::vector< std::pair< cv::Point, cv::Point> > blueRuleV;
	
};


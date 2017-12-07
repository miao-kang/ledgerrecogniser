#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/Log.h"
#include "cinder/ImageIo.h"
#include "cinder/gl/Texture.h"
#include "cinder/params/Params.h"

#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "CinderOpenCV.h"

#include "ArchivePage.h"
#include ".\nn\neuralNetwork.h"
#include ".\nn\neuralNetworkTrainer.h"

#include "windows.h"
#include <iostream>
#include <ctime>
#include <algorithm> 


using namespace std;
using namespace cv;
using namespace cv::text;

using namespace ci;
using namespace ci::app;
using namespace std;


class ArchiveParserApp : public App
{

public:

	void setup() override;

	// Mouse events
	void mouseDown(MouseEvent event) override;
	void mouseUp(MouseEvent event) override;
	void mouseMove(MouseEvent event) override;
	void mouseWheel(MouseEvent event) override;
	void mouseDrag(MouseEvent event) override;
	void keyDown(KeyEvent event) override;

	void update() override;
	void draw() override;
	void resize();

	// Buttons
	void buttonLoad();
	void buttonProcess();
	void buttonRunNNTraining(); 
	void buttonRunSingleRecognition();
	void buttonRunAreaRecognition();
	void buttonSaveBillsToCSV();

	// Image loading and processing functions
	void loadArchivePage();
	void processImageBar(Rect barRect);
	int processOneDigit(Rect oneRect, Mat tempDisplayImage, Mat tempProcessedImage, Mat temp28x28);
	Point mapSreenToImagePoint(Point screenPoint);
	Point mapImageToScreenPoint(Point imagePoint);


	// Digit classification related buttons
	void runNNTraining();
	void saveBillsToCSV();
	void runSingleRecognition();
	void runAreaRecognition();
	int classifyDigit(cv::Mat img);


private:

	gl::TextureRef		mTexture;
	gl::Texture2dRef	mProcessedImageTex;
	gl::TextureRef		mOutputTexture;
	
	cv::Mat inputImage;
	cv::Mat inputGrayImage;
	cv::Mat outputImage;

	string resultString;
	int columnNumber;

	ArchivePage pageDoc;

	// UI parameters
	CameraPersp				mCam;
	params::InterfaceGlRef	mParams;
	params::InterfaceGlRef	mParamsDigit;
	params::InterfaceGlRef	mParamsLine;
	params::InterfaceGlRef	mParamsRecognition;
	params::InterfaceGlRef	mParamsResults;

	float					mObjSize;
	quat					mObjOrientation;
	ColorA					mColor;
	string					mString;
	bool					mPrintFps;

	void					setLightDirection(vec3 direction);
	vec3					getLightDirection() { return mLightDirection; }
	vec3					mLightDirection;
	uint32_t				mSomeValue;

	vector<Point> mDetectedDigits;

	Rectf mROI;

	int counter;
	Rect mSubRect;
	Rect mSubFocusRect;

	Rectf mImageDisplayRect;

	Rectf mImageSubViewRect;

	// Mouse Handling
	Point mMouseDownPt;
	Point mMouseUpPt;

	Point  mMouseTranslate;

	bool mMouseDown;

	float mZoom;
	Point mImageViewAnchor;
	Point mImageViewMove;

	Point mTL;
	Point mBR;

	int mLineDrawing;
	Point mLineA, mLineB;

	vector<Point> mPageRules;
	PolyLine2f mConvexHull;

	// This is the digit detection parameters, Normalised height and width
	int mDigitNormalizedWidth;
	int mDigitNormalizedHeight;

	// Threshold used to get a binary image out of a grayscale image
	int mDigitThresholdMin;
	int mDigitThresholdMax;

	// Actual height and width selected from image file
	int mDigitSampleHeight;
	int mDigitSampleWidth;
	int mLineNormalizedDigitSpacing;

	// Neural Network trained with BoE ledger image digits
	int inputNN;
	int hiddenNN;
	int outputNN;
	int trainingEpochs;
	int accuracyPerCent;
};

// Prepare settings
void prepareSettings(App::Settings *settings)
{
	settings->setHighDensityDisplayEnabled();
}

// Clamp a value between a range: return min if x is less than min and return max if x is greater than max, other wise return x
template <class T>	inline T clamp(T x, T min, T max)
{
	return (x < min) ? min : (x > max) ? max : x;
}


// Initialise NN trained with BoE ledger digits, its input, hidden and output numbers to initialise
neuralNetwork nn(784, 20, 10);

// Sort contours from left to right as by default contours save objects from bottom up
struct contour_sorter_by_y
{
	bool operator ()(const vector<Point>& a, const vector<Point> & b)
	{
		// boundingRect returns topleft point of a bounding rectangle
		Rect ra(boundingRect(a));
		Rect rb(boundingRect(b));

		return std::tie(rb.y, ra.x) > std::tie(ra.y, rb.x);
	}
};

// Sort contours from top to bottom as by default contours save objects from bottom up
struct contour_sorter_by_x
{
	bool operator ()(const vector<Point>& a, const vector<Point> & b)
	{
		// boundingRect returns topleft point of a bounding rectangle
		Rect ra(boundingRect(a));
		Rect rb(boundingRect(b));

		return (ra.x < rb.x);
	}
};

// Interface and parameters initialisation
void ArchiveParserApp::setup()
{
	// Detected digit image related parameters
	mDigitNormalizedWidth = DIGIT_NORMALISED_WIDTH;
	mDigitNormalizedHeight = DIGIT_NORMALISED_HEIGHT;
	mDigitThresholdMin = DIGIT_THRESHOLD_MIN;
	mDigitThresholdMax = DIGIT_THRESHOLD_MAX;
	mDigitSampleHeight = DIGIT_SAMPLE_HEIGHT;
	mDigitSampleWidth = DIGIT_SAMPLE_WIDTH;

	// Mouse setup
	mMouseDown = false;
	counter = 0;
	mLineDrawing = 0;
	mLineA = mLineB = Point(0, 0);

	// Image viewing parameters 
	mZoom = 1.f;
	mImageViewAnchor = mImageViewMove = Point(0, 0);
	mImageDisplayRect = Rectf(250, 20, 800, 800);
	mObjSize = 125;
	mLightDirection = vec3(0, 0, -1);
	mColor = ColorA(0.25f, 0.5f, 1, 1);
	mSomeValue = 2;
	mPrintFps = false;

	// Setup default camera, looking down the z-axis.
	mCam.lookAt(vec3(-20, 0, 0), vec3(0));

	// Create the interface and their names
	mParams = params::InterfaceGl::create(getWindow(), "Document", toPixels(ivec2(200, 60)), ColorA(0.6, 0.3, 0.3, 1.f));
	mParamsRecognition = params::InterfaceGl::create(getWindow(), "Run Digit Recognition", toPixels(ivec2(200, 100)), ColorA(0.3, 0.3, 0.6, 1.f));
	mParamsResults = params::InterfaceGl::create(getWindow(), "Results", toPixels(ivec2(200, 60)), ColorA(0.2, 0.3, 0.4, 1.f));

	// Simple button under "Document" panel
	mParams->setPosition(vec2(20, 20));
	mParams->addButton("Load Document..", bind(&ArchiveParserApp::buttonLoad, this));
	mParams->addSeparator();

	function<void(vec3)> setter = bind(&ArchiveParserApp::setLightDirection, this, placeholders::_1);
	function<vec3()> getter = bind(&ArchiveParserApp::getLightDirection, this);

	// Add buttons to "Run Digit Recognition" panel
	mParamsRecognition->setPosition(vec2(20, 100));
	mParamsRecognition->addButton("Single digit recognition", bind(&ArchiveParserApp::buttonRunSingleRecognition, this));
	mParamsRecognition->addButton("Area recognition", bind(&ArchiveParserApp::buttonRunAreaRecognition, this));
	mParamsRecognition->addButton("Training neural network", bind(&ArchiveParserApp::buttonRunNNTraining, this));

	// Add buttons to "Results" panel
	mParamsResults->setPosition(vec2(20, 220));
	mParamsResults->addButton("Save results to CSV", bind(&ArchiveParserApp::buttonSaveBillsToCSV, this));

	// Parameter to save recognised results in to a CSV file
	resultString = "No of Bills,Rate percent, Discounters, Bills brought in, Bills rejected, No of Bills rejected\n";
	columnNumber = 4;

	// Set the GUI starting position and size
	setWindowPos(100, 100);
	setWindowSize(1400, 900);

	// Open a window for debugging stuff and error output
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);
	

	// Load the trained weights from neural network trained with BoE ledger image digits
	nn.loadWeights("C:\\Users\\Internet\\Desktop\\LOLR pattern recognition\\DigitRecogniser Git\\src\\nn\\weights.csv");
	trainingEpochs = TRAINING_EPOCHS;
	accuracyPerCent = ACCURACY;
	inputNN = INPUT_NEURONS;
	hiddenNN = HIDDEN_NEURONS;
	outputNN = OUTPUT_NEURONS;

}

// Map the selected image to the small window, take in a point from screen and return the new point position
Point ArchiveParserApp::mapSreenToImagePoint(cv::Point screenPoint)
{
	float fW = mZoom * (float)mOutputTexture->getWidth();
	float fH = mZoom * (float)mOutputTexture->getHeight();
	float fRW = (float)mImageDisplayRect.getWidth();
	float fRH = (float)mImageDisplayRect.getHeight();
	float fX1 = (float)(screenPoint.x - mImageDisplayRect.x1);
	float fY1 = (float)(screenPoint.y - mImageDisplayRect.y1);

	float px = fW*(fX1 / fRW);
	float py = fH*(fY1 / fRH);

	return Point(int(px), int(py));
}

// Map the image from small window to screen, take in a point from small window and return the new point position in screen
Point ArchiveParserApp::mapImageToScreenPoint(cv::Point imagePoint)
{
	// Draw some ROI blow up 
	float fW = (float)mOutputTexture->getWidth();
	float fH = (float)mOutputTexture->getHeight();
	float fRW = (float)mImageDisplayRect.getWidth();
	float fRH = (float)mImageDisplayRect.getHeight();
	float fX1 = (float)(imagePoint.x);
	float fY1 = (float)(imagePoint.y);

	float px = mImageDisplayRect.x1 + fRW*(fX1 / fW);
	float py = mImageDisplayRect.x1 + fRH*(fY1 / fH);

	return Point(int(px), int(py));
}

// Load the image user tries to open, throw an error is file is not found
void ArchiveParserApp::loadArchivePage()
{
	try
	{
		fs::path path = getOpenFilePath("", ImageIo::getLoadExtensions());
		if (!path.empty())
		{
			mTexture = gl::Texture::create(loadImage(path));
			Surface processedImage(loadImage(path));
			mProcessedImageTex = gl::Texture2d::create(processedImage);

			ci::Surface8u surface(processedImage);

			inputImage = toOcv(processedImage);

			outputImage = inputImage.clone();

			mOutputTexture = gl::Texture::create(fromOcv(outputImage));

			mParams->setOptions("DocName", path.generic_string());

			mTL = Point(0, 0);
			mBR = Point(inputImage.rows, inputImage.cols);

			mImageSubViewRect = Rectf(0, 0, inputImage.rows, inputImage.cols);

		}
	}
	catch (cinder::Exception &exc)
	{
		CI_LOG_EXCEPTION("failed to load image.", exc);
	}
}

// Button to call method which will re-train the neural network
void ArchiveParserApp::buttonRunNNTraining()
{
	runNNTraining();
}

// Button to call method which will recognise the single digit selected, using trained neural network
void ArchiveParserApp::buttonRunSingleRecognition()
{
	runSingleRecognition();
}

// Button to call method which will recognise the digits with area selected, using trained neural network
void ArchiveParserApp::buttonRunAreaRecognition()
{
	runAreaRecognition();
}

// Button to call method which will save recognised digits to a CSV file
void ArchiveParserApp::buttonSaveBillsToCSV()
{
	saveBillsToCSV();
}

// Button to load an archival image file
void ArchiveParserApp::buttonLoad()
{
	loadArchivePage();
}

// Button to process image
void ArchiveParserApp::buttonProcess()
{
	cv::Mat bgr[3];
	cv::split(inputImage, bgr);

	equalizeHist(bgr[0], bgr[0]);
	equalizeHist(bgr[1], bgr[1]);
	equalizeHist(bgr[2], bgr[2]);

	cv::merge(bgr, 3, outputImage);

	mOutputTexture = gl::Texture::create(fromOcv(outputImage));
}

// Set the light direction to view image file
void ArchiveParserApp::setLightDirection(glm::vec3 direction)
{
	console() << "Light direction: " << direction << endl;
	mLightDirection = direction;
}

// Resize
void ArchiveParserApp::resize()
{
	mCam.setAspectRatio(getWindowAspectRatio());
}

// Update
void ArchiveParserApp::update()
{
	if (mPrintFps && getElapsedFrames() % 60 == 0)
		console() << getAverageFps() << endl;
}

// Actions when mouse is clicked
void ArchiveParserApp::mouseDown(cinder::app::MouseEvent event)
{
	mMouseDownPt.x = event.getPos().x;
	mMouseDownPt.y = event.getPos().y;


	mMouseDown = true;
	mLineA = mMouseDownPt;

	// Image view control
	if (event.isRight())
	{
		mImageViewAnchor = mapSreenToImagePoint(mMouseDownPt);
	}

}

// Actions when mouse is moved
void ArchiveParserApp::mouseMove(cinder::app::MouseEvent event)
{
		
}

// Actions when mouse is dragged
void ArchiveParserApp::mouseDrag(cinder::app::MouseEvent event)
{
	Point mousePoint(event.getPos().x, event.getPos().y);

	// Actions when mouse is right clicked
	if (event.isRightDown())
	{
		mImageViewMove = mapImageToScreenPoint(mousePoint);
	}

	// Actions when mouse is left clicked
	if (event.isLeftDown() && (mLineDrawing == 1))
	{
		mLineB.x = event.getPos().x;
		mLineB.y = event.getPos().y;
	}

}

// Actions when mouse is up
void ArchiveParserApp::mouseUp(cinder::app::MouseEvent event)
{

	mMouseDown = false;

	mMouseUpPt.x = event.getPos().x;
	mMouseUpPt.y = event.getPos().y;

	mLineB = mMouseUpPt;

	// Reset movement
	if (event.isRight())
	{
		mImageViewMove = Point(0, 0);
	}

}

// Actions when mouse is down
void ArchiveParserApp::keyDown(cinder::app::KeyEvent event)
{

}

// Actions when mouse wheel is used to zoom in or out
void ArchiveParserApp::mouseWheel(cinder::app::MouseEvent event)
{
	float mouseDelta = event.getWheelIncrement() *0.1f;

	mZoom += mouseDelta;

	mZoom = clamp(mZoom, 1.f, 20.f);

	mImageSubViewRect.x2 = float(inputImage.cols) / mZoom;
	mImageSubViewRect.y2 = float(inputImage.rows) / mZoom;

}

// Method to run a single digit recognition
void ArchiveParserApp::runSingleRecognition()
{
	// Create an image using the selected area
	Rect subRect;
	Mat subImage = inputImage(mSubRect);
	cv::imwrite(".\\testSubImage.png", subImage);

	// Create a temp image
	Mat tempGrey, temp28x28(28, 28, CV_8UC1);

	// Convert the input image from RGB to gray
	cvtColor(subImage, tempGrey, CV_RGB2GRAY);
	cv::imwrite(".\\testSubImageGray.png", tempGrey);

	// Convert to binary image
	Mat tempProc = tempGrey;
	tempProc = cv::Scalar::all(255) - tempProc;
	// The saved png image will help developers to debug
	cv::imwrite(".\\testSubImageGrayInv.png", tempGrey);

	cv::threshold(tempProc, tempProc, mDigitThresholdMin, mDigitThresholdMax, THRESH_TOZERO);
	// The saved png image will help developers to debug
	cv::imwrite(".\\testSubImageGrayInvThresh.png", tempProc);

	Mat tempProcessedImage = tempProc.clone();
	Mat tempDisplayImage = tempProc.clone();

	Moments m = moments(tempProc, false);
	Point2d p(m.m10 / m.m00, m.m01 / m.m00);

	// Convex hull vectors
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// Find the contours which bind the digit object
	cv::findContours(tempProc, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	// Contours parameters initialisation
	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;

	// Find the convex hull object for each contour
	vector<vector<Point>> hull(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		// Find the area of contour
		double a = contourArea(contours[i], false);
		if (a > largest_area)
		{
			largest_area = a;
			// Store the index of largest contour
			largest_contour_index = i;
			// Find the bounding rectangle for biggest contour
			bounding_rect = boundingRect(contours[i]);
		}
	}

	// Extract the largest 
	cv::imwrite(".\\testSubImageGrayInvThreshAfterContours.png", tempProcessedImage);

	Mat boxImg = tempProcessedImage(bounding_rect);
	cv::imwrite(".\\testSubImageGrayInvThreshProc.png", boxImg);

	Mat temp20x20(mDigitSampleWidth, mDigitSampleHeight, CV_8UC1);
	cv::resize(boxImg, temp20x20, temp20x20.size());

	// Copy to inset region
	temp28x28.setTo(0);
	temp20x20.copyTo(temp28x28(Rect(3, 3, temp20x20.rows, temp20x20.cols)));

	cv::imwrite(".\\testSubImageGrayInvThreshProcFinal.png", temp28x28);
	cv::imshow("tempwin", temp28x28);

	// Classify using NN trained with BoE ledger digit images
	int digitFoundNew = classifyDigit(temp28x28);

	cout << "Recognised digits " << digitFoundNew << endl;

	mDetectedDigits.clear();
	mDetectedDigits.push_back(Point(bounding_rect.x, digitFoundNew));

}

// Function to call area recognition method
void ArchiveParserApp::runAreaRecognition()
{
	processImageBar(mSubFocusRect);
}

// Process and recognise one digit seperated from large contour in area recognition
int ArchiveParserApp::processOneDigit(cv::Rect oneRect, cv::Mat tempDisplayImage, cv::Mat tempProcessedImage, cv::Mat temp28x28)
{
	// Create new iamge from image separated from large contour
	cv::rectangle(tempDisplayImage, oneRect, cv::Scalar(200, 0, 0));
	Mat boxImg1 = tempProcessedImage(oneRect);

	// The saved png image will help developers to debug
	cv::imwrite(".\\testSubImageGrayInvThreshProc.png", boxImg1);
	Mat tempLeft20x20(20, 20, CV_8UC1);
	cv::resize(boxImg1, tempLeft20x20, tempLeft20x20.size());
	temp28x28.setTo(0);
	tempLeft20x20.copyTo(temp28x28(Rect(3, 3, tempLeft20x20.rows, tempLeft20x20.cols)));

	// The saved png image will help developers to debug
	cv::imwrite(".\\testSubImageGrayInvThreshProcFinal.png", temp28x28);

	// Use neural network to classify the digit
	int digitFoundNewOne = classifyDigit(temp28x28);
	return digitFoundNewOne;
}

// Method to run area recognition, by processing the image, detecting all digit objects, arranging them in the correct order and feeding them to NN for recognition
void ArchiveParserApp::processImageBar(cv::Rect barRect)
{
	// Create an image using the selected area
	Mat subImage = inputImage(barRect);
	// The saved png image will help developers to debug
	cv::imwrite(".\\testSubImage.png", subImage);

	// Create a temp image
	Mat tempGrey, temp28x28(28, 28, CV_8UC1);

	// Convert the input image from RGB to gray
	cvtColor(subImage, tempGrey, CV_RGB2GRAY);
	// The saved png image will help developers to debug
	cv::imwrite(".\\testSubImageGray.png", tempGrey);

	// Convert to binary image
	Mat tempProc = tempGrey;
	tempProc = cv::Scalar::all(255) - tempProc;
	// The saved png image will help developers to debug
	cv::imwrite(".\\testSubImageGrayInv.png", tempGrey);

	cv::threshold(tempProc, tempProc, 140, 255, THRESH_TOZERO);
	// The saved png image will help developers to debug
	cv::imwrite(".\\testSubImageGrayInvThresh.png", tempProc);

	
	Mat tempProcessedImage = tempProc.clone();
	Mat tempDisplayImage = tempProc.clone();

	// Get the convex hull
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// find the contours which bind the digit objects
	cv::findContours(tempProc, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// Initialise parameters
	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect, bounding1, bounding2;
	mDetectedDigits.clear();

	// Find the convex hull object for each contour
	cout << "Digits: " << endl;
	vector<vector<Point>> hull(contours.size());


	// Remove small contours which has less than 10 pixels
	for (vector<vector<Point> >::iterator it = contours.begin(); it != contours.end();)
	{
		if (it->size()<10)
			it = contours.erase(it);
		else
			++it;
	}

	// Sort contours to start from left to right and top to bottom
	std::sort(contours.begin(), contours.end(), contour_sorter_by_y());

	// Find line breaks
	vector<pair<int, int>> lineBreaks;

	int start = 0, end = 0;
	for (int i = start; i < contours.size() - 1; i++)
	{
		bounding1 = boundingRect(contours[i]);
		bounding2 = boundingRect(contours[i + 1]);
		if ((bounding2.y - bounding1.y) > 25)
		{
			end = i;
			lineBreaks.push_back(std::make_pair(start, end));
			start = end + 1;
		}

	}

	// Separate contours line by line 
	for (std::vector<pair<int, int>>::iterator it = lineBreaks.begin(); it != lineBreaks.end(); ++it)
	{
		int lineStart = it->first;
		int lineEnd = it->second;
		int digitRecognised = 0;
		char tempNumberLeft[10];

		// Create vector to store contours by lines
		vector< vector<Point> > contoursByLine;
		for (int i = lineStart; i < lineEnd + 1; i++)
			contoursByLine.push_back(contours[i]);

		// Sort the contours by left to right and top to down
		std::sort(contoursByLine.begin(), contoursByLine.end(), contour_sorter_by_x());

		// Result string to write to the 
		string lineString = "";

		for (int i = 0; i < contoursByLine.size(); i++)
		{
			//  Find the area of contour
			double a = contourArea(contoursByLine[i], false);  
			// Find the height of contour
			double h = boundingRect(contoursByLine[i]).height; 
			// Find the width of contour
			double w = boundingRect(contoursByLine[i]).width; 

			if ((h<70) && (w>20) && (h>20))
			{
				// If the detected contour is about three digits' width, then split this contour equally to three contours
				if (w > 85)
				{
					// Find the bounding rectangle for biggest contour, divide width by 3
					bounding_rect = boundingRect(contoursByLine[i]); 
					float f = (float)1 / 3;
					Rect leftRect(bounding_rect.x, bounding_rect.y, f*w, h);
					Rect middleRect(bounding_rect.x + f*w, bounding_rect.y, f*w, h);
					Rect rightRect(bounding_rect.x + f * 2 * w, bounding_rect.y, f*w, h);

					// Process and recognise the left one
					digitRecognised = processOneDigit(leftRect, tempDisplayImage, tempProcessedImage, temp28x28);
					lineString += std::to_string(digitRecognised);
					cout << digitRecognised;
					 sprintf(tempNumberLeft, "%d", digitRecognised);
					cv::putText(tempDisplayImage, tempNumberLeft, leftRect.tl(), FONT_HERSHEY_SIMPLEX, 10, cv::Scalar(0, 0, 0));
					mDetectedDigits.push_back(Point(leftRect.x, digitRecognised));

					// Process and recognise the middle one
					digitRecognised = processOneDigit(middleRect, tempDisplayImage, tempProcessedImage, temp28x28);
					lineString += std::to_string(digitRecognised);
					cout << digitRecognised;
					sprintf(tempNumberLeft, "%d", digitRecognised);
					cv::putText(tempDisplayImage, tempNumberLeft, middleRect.tl(), FONT_HERSHEY_SIMPLEX, 10, cv::Scalar(0, 0, 0));
					mDetectedDigits.push_back(Point(middleRect.x, digitRecognised));

					// Process and recognise the right one
					digitRecognised = processOneDigit(rightRect, tempDisplayImage, tempProcessedImage, temp28x28);
					lineString += std::to_string(digitRecognised);
					cout << digitRecognised;
					sprintf(tempNumberLeft, "%d", digitRecognised);
					cv::putText(tempDisplayImage, tempNumberLeft, rightRect.tl(), FONT_HERSHEY_SIMPLEX, 10, cv::Scalar(0, 0, 0));
					mDetectedDigits.push_back(Point(rightRect.x, digitRecognised));

				}
				// Otherwise if the detected contour is about two digits' width, then split this contour equally to two contours
				else if (w > 45)
				{
					// Find the bounding rectangle for biggest contour
					bounding_rect = boundingRect(contoursByLine[i]); 
					// Split the two digits
					Rect leftRect(bounding_rect.x, bounding_rect.y, 0.5*w, h);
					Rect rightRect(bounding_rect.x + 0.5*w, bounding_rect.y, 0.5*w, h);

					// Process and recognise the left one
					digitRecognised = processOneDigit(leftRect, tempDisplayImage, tempProcessedImage, temp28x28);
					lineString += std::to_string(digitRecognised);
					cout << digitRecognised;
					sprintf(tempNumberLeft, "%d", digitRecognised);
					cv::putText(tempDisplayImage, tempNumberLeft, leftRect.tl(), FONT_HERSHEY_SIMPLEX, 10, cv::Scalar(0, 0, 0));
					mDetectedDigits.push_back(Point(leftRect.x, digitRecognised));

					// Process and recognise the right one
					digitRecognised = processOneDigit(rightRect, tempDisplayImage, tempProcessedImage, temp28x28);
					lineString += std::to_string(digitRecognised);
					cout << digitRecognised;
					sprintf(tempNumberLeft, "%d", digitRecognised);
					cv::putText(tempDisplayImage, tempNumberLeft, rightRect.tl(), FONT_HERSHEY_SIMPLEX, 10, cv::Scalar(0, 0, 0));
					mDetectedDigits.push_back(Point(rightRect.x, digitRecognised));

				}
				// Otherwise the digit image has the correct size
				else
				{
					// Find the bounding rectangle for biggest contour
					bounding_rect = boundingRect(contoursByLine[i]); 
					cv::rectangle(tempDisplayImage, bounding_rect, cv::Scalar(200, 0, 0));
					Mat boxImg = tempProcessedImage(bounding_rect);

					cv::imwrite(".\\testSubImageGrayInvThreshProc.png", boxImg);
					Mat temp20x20(20, 20, CV_8UC1);
					cv::resize(boxImg, temp20x20, temp20x20.size());

					// Copy to inset region
					temp28x28.setTo(0);
					temp20x20.copyTo(temp28x28(Rect(3, 3, temp20x20.rows, temp20x20.cols)));

					cv::imwrite(".\\testSubImageGrayInvThreshProcFinal.png", temp28x28);

					// NN trained with BoE ledger digits
					digitRecognised = classifyDigit(temp28x28);
					lineString += std::to_string(digitRecognised);
					cout << digitRecognised;

					char tempNumber[10]; sprintf(tempNumber, "%d", digitRecognised);
					cv::putText(tempDisplayImage, tempNumber, bounding_rect.tl(), FONT_HERSHEY_SIMPLEX, 10, cv::Scalar(0, 0, 0));

					mDetectedDigits.push_back(Point(bounding_rect.x, digitRecognised));
				}

			}
		}
		
		// Insert the detected numbers for each line to the correct column, this will be writtent to the result csv file
		switch (columnNumber)
		{
		case 1:
			resultString += lineString + ",,,,,\n";
			break;
		case 2:
			resultString += "," + lineString + ",,,,\n";
			break;
		case 3:
			resultString += ",," + lineString + ",,,\n";
			break;
		case 4:
			resultString += ",,," + lineString + ",,\n";
			break;
		case 5:
			resultString += ",,,," + lineString + ",\n";
			break;
		case 6:
			resultString += ",,,,," + lineString + "\n";
			break;
		default:
			cout << "Invalid column" << endl;
		}

		cout << " " << endl;
	}

	cout << endl;

	cv::imshow("tempwin", tempDisplayImage);
	cv::imwrite(".\\testSubImageGrayInvThreshProcFinalOverlay.png", tempDisplayImage);

	MessageBox(
		NULL,
		(LPCWSTR)L"Recognition completed, please check console for results or\nclick on 'Save results to CSV' button to save results to CSV file!",
		(LPCWSTR)L"Successfully recognised",
		MB_OK);

}

// Method to save recognition results to CSV files
void ArchiveParserApp::saveBillsToCSV()
{
	// Open file for reading
	fstream outputFile;
	outputFile.open(".\\results.csv", ios::out);

	if (outputFile.is_open())
	{
		outputFile.precision(50);
		outputFile << resultString;

		// Print success - debugging purposes
		cout << endl << "Results saved " << endl;

		// Close file
		outputFile.close();

		MessageBox(
			NULL,
			(LPCWSTR)L"Results are saved in results.csv file!",
			(LPCWSTR)L"Successfully saved",
			MB_OK);
	}
	else
	{
		cout << endl << "Error - results file could not be created: " << endl;
	}

}

// Re-train the NN using BoE ledger digits. The saved weight.csv file contains the trained neural network so it doesn't have to be re-trained each run.
// However, if you'd want to change NN parameters or have produced more training data, then use this function to re-train the neural network
void ArchiveParserApp::runNNTraining()
{
	// Random number generator
	srand((unsigned int)time(0));

	// Create data set reader and training data file
	dataReader d;
	d.loadDataFile("..\\src\\nn\\ledgerdigits.csv", inputNN, outputNN);
	d.setCreationApproach(STATIC, 10);

	// Create neural network
	neuralNetwork nn(inputNN, hiddenNN, outputNN);

	// Create neural network trainer
	neuralNetworkTrainer nT(&nn);
	nT.setTrainingParameters(0.001, 0.9, false);
	nT.setStoppingConditions(trainingEpochs, accuracyPerCent);
	nT.enableLogging("..\\src\\nn\\log.csv", 5);

	// Train neural network on data sets
	for (int i = 0; i < d.getNumTrainingSets(); i++)
	{
		nT.trainNetwork(d.getTrainingDataSet());
	}

	// Save the learned weights
	nn.saveWeights("..\\src\\nn\\weights.csv");

	cout << endl << endl << "-- END OF PROGRAM --" << endl;
	char c; cin >> c;
}

// Concatenate images
Mat concatenateMatt(std::vector<cv::Mat> &vec)
{
	int height = vec[0].rows;
	int width = vec[0].cols;
	Mat res = Mat::zeros(height * width, vec.size(), CV_64FC1);
	for (int i = 0; i<vec.size(); i++)
	{
		Mat img(height, width, CV_64FC1);

		vec[i].convertTo(img, CV_64FC1);
		Mat ptmat = img.reshape(0, height * width);
		Rect roi = cv::Rect(i, 0, ptmat.cols, ptmat.rows);
		Mat subView = res(roi);
		ptmat.copyTo(subView);
	}
	divide(res, 255.0, res);
	return res;
}

// Method used to take in processed contour and feed in to the neural network for classification
int ArchiveParserApp::classifyDigit(cv::Mat img)
{
	std::vector<Mat> vimg;
	vimg.push_back(img);
	Mat tx = concatenateMatt(vimg);

	// Generate input from the image to the trained neural network
	double* p = new double[inputNN];

	for (int i = 0; i < tx.rows; i++)
	{
		p[i] = tx.ptr<double>(i)[0];
	}

	int* result;
	int digitFound = -1;

	// Feed the input of image to neurnetwork and get a recognition result
	result = nn.feedForwardPattern(p);
	for (int i = 0; i < outputNN; i++)
	{
		// If the neural network produced a positive output (1), then learned digit is found
		if (result[i] == 1)
			digitFound = i + 1;
		// If the result is 10, then the digit is 0 (training pattern is arranged as 1, 2...0
		if (digitFound == 10)
			digitFound = 0;
	}

	// If none of the classes has produced 1, then the winning class is the one with highest output
	if (digitFound == -1)
	{
		int maxIndex = 0;
		int maxValue = result[0];
		for (int i = 1; i < outputNN; i++)
		{
			if (maxValue < result[i])
			{
				maxIndex = i;
				maxValue = result[i];
			}
		}

		digitFound = maxIndex;
	}

	return digitFound;
}


// Draw results to interface
void ArchiveParserApp::draw()
{

	gl::clear(Color(0.5f, 0.5f, 0.5f));
	gl::enableAlphaBlending();


	if (mOutputTexture)
	{
		// This draws the output processed image.
		Rectf destRect(250, 20, 800, 850); 

		// Draw some ROI blow up 
		float fW = (float)mOutputTexture->getWidth();
		float fH = (float)mOutputTexture->getHeight();
		float fRW = (float)destRect.getWidth();
		float fRH = (float)destRect.getHeight();
		float fX1 = (float)(mMouseDownPt.x - destRect.x1);
		float fY1 = (float)(mMouseDownPt.y - destRect.y1);
		float fX2 = (float)(mMouseUpPt.x - destRect.x1);
		float fY2 = (float)(mMouseUpPt.y - destRect.y1);

		Area focus;
		focus.x1 = fW*(fX1 / fRW);
		focus.y1 = fH*(fY1 / fRH);
		focus.x2 = fW*(fX2 / fRW);
		focus.y2 = fH*(fY2 / fRH);

		int dY = mDigitNormalizedHeight, dX = mDigitNormalizedWidth;
		Area miniFocus;
		miniFocus = focus;
		miniFocus.x1 -= dX / 2;
		miniFocus.y1 -= dY / 2;
		miniFocus.x2 = miniFocus.x1 + dX;
		miniFocus.y2 = miniFocus.y1 + dY;

		mSubFocusRect.x = focus.x1;
		mSubFocusRect.y = focus.y1;
		mSubFocusRect.width = focus.x2 - focus.x1;
		mSubFocusRect.height = focus.y2 - focus.y1;

		gl::draw(mOutputTexture, focus, Rectf(1100, 300, 1400, 500));

		// Draw the entire image with zoom and translation
		Area imageAreaToDraw;

		// Work out some image drag information (in image coordinates);
		int imageDragX = (mImageViewAnchor.x - mImageViewMove.x); 
		int imageDragY = (mImageViewAnchor.y - mImageViewMove.y); 

		imageAreaToDraw.x1 = clamp(imageDragX, 0, (int)mImageSubViewRect.getWidth());
		imageAreaToDraw.x2 = clamp(int(mImageSubViewRect.x2), 0, inputImage.cols);
		imageAreaToDraw.y1 = clamp(imageDragY, 0, (int)mImageSubViewRect.getHeight());
		imageAreaToDraw.y2 = clamp(int(mImageSubViewRect.y2), 0, inputImage.rows);

		mImageSubViewRect.x1 = imageAreaToDraw.x1;
		mImageSubViewRect.x2 = imageAreaToDraw.x2;
		mImageSubViewRect.y1 = imageAreaToDraw.y1;
		mImageSubViewRect.y2 = imageAreaToDraw.y2;

		gl::draw(mOutputTexture, imageAreaToDraw, destRect);

		Font font = Font("Arial", 28);

		for (int i = 0; i < mDetectedDigits.size(); i++)
		{
			char temp[10];
			sprintf(temp, "%d", mDetectedDigits[i].y);
			gl::drawString(temp, vec2(mDetectedDigits[i].x + 1100, 300), ColorA(0.1, 0.1, 0.1, 10.f), font);
		}

		mSubRect.x = miniFocus.x1;
		mSubRect.y = miniFocus.y1;
		mSubRect.width = dX;
		mSubRect.height = dY;

		gl::draw(mOutputTexture, miniFocus, Rectf(1100, 500, 1400, 700));
	}

	gl::color(1.f, 0.f, 0.f, 0.5f);
	gl::drawStrokedRect(Rectf(mMouseDownPt.x, mMouseDownPt.y, mMouseUpPt.x, mMouseUpPt.y), 1);

	// Draw the document rules and template
	gl::color(1.f, 1.f, 1.f, 1.0f);
	if (mLineDrawing > 0)
	{
		gl::drawLine(vec2(mLineA.x, mLineA.y), vec2(mLineB.x, mLineB.y));

		gl::color(0.f, 0.f, 1.f, 1.f);
		for (int i = 0; i < pageDoc.blueRuleV.size(); i++)
		{
			gl::drawLine(vec2(pageDoc.blueRuleV[i].first.x, pageDoc.blueRuleV[i].first.y), vec2(pageDoc.blueRuleV[i].second.x, pageDoc.blueRuleV[i].second.y));
		}

		for (int i = 0; i < pageDoc.blueRuleH.size(); i++)
		{
			gl::drawLine(vec2(pageDoc.blueRuleH[i].first.x, pageDoc.blueRuleH[i].first.y), vec2(pageDoc.blueRuleH[i].second.x, pageDoc.blueRuleH[i].second.y));
		}

		gl::color(1.f, 0.f, 0.f, 1.f);
		for (int i = 0; i < pageDoc.redRuleV.size(); i++)
		{
			gl::drawLine(vec2(pageDoc.redRuleV[i].first.x, pageDoc.redRuleV[i].first.y), vec2(pageDoc.redRuleV[i].second.x, pageDoc.redRuleV[i].second.y));
		}

		gl::color(1.f, 1.f, 1.f, 1.f);
	}

	// Draw the interface
	mParams->draw();
}

CINDER_APP(ArchiveParserApp, RendererGl)

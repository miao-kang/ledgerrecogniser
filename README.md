# Digits Recognition Tool for Bank of England 19th Century Archival Discount Ledgers

## 1. Libraries and Setup Details

<p align="justify">The core image loading/processing and saving, as well as, auxiliary functionality is implemented by using the library: OpenCV 3.1 - http://opencv.org/ 
The user interface and some of the directory/file management is done through the library Cinder 0.9 - https://libcinder.org/ 
Cinder comes with an installation tool which also allows OpenCV installation, however, to be using the latest OpenCV version it is best to install separately.
OpenCV provides the main functionality for working with images of archive pages. We use it to pre-process data, crop region, perform scaling and normalisation operations to get the data into a form where it can be passed onto the main recognition class.
</p>
## 2. Neural Network Recognition Tool:

<p align="justify">The recognition tool is based on C++ implementation (suing OpenCV helper functions) and built in with a simple feed forward neural network trained with data extracted from digitised Bank of England archival discount ledger images directly. </p>  

### 2.1 Feed-forward Neural Network:

<p align="justify">This simple feed-forward neural network is adapted from open source code project here. It has a structure of 784 x 20 x10 which can be changed in ArchiveParser.h file. The training digit images have been normalised to 28 x 28 pixels hence a total number of 784 inputs. 10 outputs are simply the 10 digits being trained (1, 2… to 0). We only tested on one layer of hidden neurons and found 20 hidden neurons produce relatively the better performance.
Once trained, the learned weights are stored in weights.csv file which will be loaded automatically by the recognition tool and used in real time recognition when being initialised when app starts. </p>  

### 2.2 Training Data:

<p align="justify">The digits training data are extracted from digitised Bank of England archival discount ledger files (as shown in Figure 1):</p>  
<p align="center">
  <img src="https://github.com/boeml/ledgerrecogniser/blob/master/readmeimages/figure%201.jpg">
  <br>Figure 1. Digitised Bank of England Archival Discount Ledger File
</p>

<p align="justify">70 samples are randomly selected for each digit from these ledger files across 1847 to 1914, hence a total number of 700 training patterns have been produced. Each digit is converted to a binary image and then normalised to 28 x 28 pixels. The training data is saved in “ledgerdigits.cvs” file.</p>   
<p align="center">
  <img src="https://github.com/boeml/ledgerrecogniser/blob/master/readmeimages/figure%202.jpg">
  <br>Figure 2. Training patterns extracted from digitised BoE archival discount ledger files
</p>

<p align="justify">Due to the time constraints, we haven’t been able to produce more training patterns. We recommend that large number of training patterns will significantly improve the accuracy of the recognition. </p>  

## 3. User Interface and Recognition Tool

<p align="justify">The digit detection network is implemented as a class within a software tool that provides a user interface. The main reason for this is that detecting digits requires some manual interaction in order to guide the detector to appropriate regions and subsequently annotate the regions into useful data for analytics. 
The user interface allows loading of images and requires additional development to put it into full use and to structure the interaction of the user to the code that is embedded in the software. This is because interaction requires various levels eg. image zoom and movement, selection of image regions etc. While not complicated these take time to get right and require normalisation of coordinates/image regions to function properly.
Currently, when the interface launches in the setup function both neural networks are loaded with pre-trained weights.  
User can load a digitised discount ledger file using “Load Document” button, once loaded user can test a single digit recognition by single clicking on a digit, the tool will automatically detect the digit object, please use button “Single digit recognition” to see the recognition result. This single digit recognition function is purely for the development and research purposes as user can easily investigate how each digit is being selected and recognised. </p>  
<p align="center">
  <img src="https://github.com/boeml/ledgerrecogniser/blob/master/readmeimages/figure%203.jpg">
  <br>Figure 3. Single digit recognition 
</p>

<p align="justify">As the trained network model takes in 28 x 28 pixel sized images. Ideally the digit should be centred in the image for best chances of correct recognition. To get the raw data into this form some pre-processing is required.
The first processing task is to invert the image. The data in these ledger images are typically on white/yellow paper with a darker ink. Therefore we preform inversion by replacing each pixel value with 255 – original pixel value. The result looks like this:</p>  

<p align="center">
  <img src="https://github.com/boeml/ledgerrecogniser/blob/master/readmeimages/figure%204.jpg">
  <br>Figure 4. Input image cropped area of numbers (left) and the resulting processed region (right)
</p>

<p align="justify">Then the first operation within a candidate region is for the image to be thresholded to produce a binary image. This is perform this using the OpenCV function:
	cv::threshold(image, image, 140, 255, THRESH_TOZERO);
The resulting image will look something like this:</p>  

<p align="center">
  <img src="https://github.com/boeml/ledgerrecogniser/blob/master/readmeimages/figure%205.jpg">
  <br>Figure 5. Input image cropped area of numbers (left) and the resulting processed region (right)
</p>

<p align="justify">The values of the thresholding can be varied and would impact on the performance of the algorithm. Additionally the method of thresholding can also be changed. In a more complex implementation instead of thresholding classification algorithms could be used to perform better ink / paper segmentation and potentially introduce spatial regularisation/connectivity.
The binary image is then passed to a contour finding function, which aims to essentially find the outer contour of the digit and allow us to compute the convex hull or bounding box, which can then be used to estimate the digit centre. Again, this is performed using OpenCV through the function:
cv::findContours(image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
From the extracted contour, one is able to get the centre of the candidate digit by computing the bounding box of the individual contours:</p>  
<p align="center">
  <img src="https://github.com/boeml/ledgerrecogniser/blob/master/readmeimages/figure%206.jpg">
  <br>Figure 6. Input image cropped area of numbers (left) and the resulting processed region (right)
</p>  

<p align="justify">Then the original raw image is resampled at that point to extract a 20 x 20 pixel region. This region is copied into a blank (black) background template image of size 28 x 28. This is the template sent to the network for recognition:
	DigitRecognizer::classify(image_template_28x28);
The area recognition requires a bit of manual process as we are currently only focusing on digit recognition, user needs to select an area to be transcribed (which contains numbers only):</p>  

<p align="center">
  <img src="https://github.com/boeml/ledgerrecogniser/blob/master/readmeimages/figure%207.jpg">
  <br>Figure 7: Area recognition on discount ledger images
</p>  

<p align="justify">Algorithms have been implemented to automatically sort each digit object detected in the selected area from top to down and from left to right. Detected objects are also checked for their width and are broken down into small objects if it is out of certain range (for example the continuous 00s or 000s in figure 8).</p>  

<p align="center">
  <img src="https://github.com/boeml/ledgerrecogniser/blob/master/readmeimages/figure%208.jpg">
  <br>Figure 8. Detected digit objects within selected area
</p>  

The tool will also save recognition results to a CSV file, results will look like below:
<p align="center">
  <img src="https://github.com/boeml/ledgerrecogniser/blob/master/readmeimages/figure%209.jpg">
  <br>Figure 9. Results are saved in CSV file
</p>  

## 4. Future Development Ideas:

<p align="justify">The tool implemented/experimented some basic features of detecting and recognising hand written figures from the mid-nineteenth century Bank of England discount ledger files. We haven’t been able to develop/investigate detailed implementation of the tool due to time constrains. It however opens the door for anyone who is interested in helping us dig out more of these buried treasures and make the tool more widely usable.  
Several possibilities for further development are:
-	Make digits object detection more accurate: currently if a digit object doesn’t have enough outer/biding space, it won’t be detected as an object, i.e. if the digit object is connected to a column or row line like below:
        ![ScreenShot](https://github.com/boeml/ledgerrecogniser/blob/master/readmeimages/figure%2010.jpg)
-	Produce/use more training data: we’ve only created 700 training data so far by using the image ledgers, it would be useful to see the results produced from neural network trained with more data
-	Allow user to design template to identify regions (with lines of rows and columns) which is applicable to a batch of similar files. This will automate the manual selection process of areas to recognise and enable batch processing of such files
-	Can we extend the recognition from digits to letters (i.e. on counterparty names and other characters)?
</p>  

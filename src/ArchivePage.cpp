#include "ArchivePage.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <fstream>
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

ArchivePage::ArchivePage()
{

}

ArchivePage::~ArchivePage()
{

}

// Load a saved page template
void ArchivePage::loadPage(std::string filename)
{
	FileStorage fs(filename, FileStorage::READ);
	
	blueRuleH.clear();
	blueRuleV.clear();
	redRuleV.clear();

	FileNode brl = fs["blue rules vertical"];
	FileNodeIterator it = brl.begin(), it_end = brl.end();

	int idx = 0;
	for (; it != it_end; ++it, idx++)
	{
		Point a, b;
		a.x = (int)(*it)["x0"]; a.y = (int)(*it)["y0"];
		b.x = (int)(*it)["x1"]; b.y = (int)(*it)["y1"];
		blueRuleV.push_back(std::pair<Point,Point>(a, b));
	}

	FileNode brh = fs["blue rules horizontal"];
	it = brh.begin(), it_end = brh.end();
	for (; it != it_end; ++it, idx++)
	{
		Point a, b;
		a.x = (int)(*it)["x0"]; a.y = (int)(*it)["y0"];
		b.x = (int)(*it)["x1"]; b.y = (int)(*it)["y1"];
		blueRuleH.push_back(std::pair<Point, Point>(a, b));
	}

	FileNode rrh = fs["red rules horizontal"];
	it = rrh.begin(), it_end = rrh.end();
	for (; it != it_end; ++it, idx++)
	{
		Point a, b;
		a.x = (int)(*it)["x0"]; a.y = (int)(*it)["y0"];
		b.x = (int)(*it)["x1"]; b.y = (int)(*it)["y1"];
		redRuleV.push_back(std::pair<Point, Point>(a, b));
	}

	fs.release();
}

// Save a page template
void ArchivePage::savePage(std::string filename)
{
	FileStorage fs(filename, FileStorage::WRITE);

	time_t rawtime; time(&rawtime);
	fs << "Archive Processing Date" << asctime(localtime(&rawtime));

	// Store the template information
	fs << "blue rules vertical" << "[";
	for (int i = 0; i < blueRuleV.size(); i++)
	{
		fs << "{:" << "x0" << blueRuleV[i].first.x << "y0" << blueRuleV[i].first.y << "x1" << blueRuleV[i].second.x << "y1" << blueRuleV[i].second.y << "}";
	}
	fs << "]";

	// Store the template information
	fs << "blue rules horizontal" << "[";
	for (int i = 0; i < blueRuleH.size(); i++)
	{
		fs << "{:" << "x0" << blueRuleH[i].first.x << "y0" << blueRuleH[i].first.y << "x1" << blueRuleH[i].second.x << "y1" << blueRuleH[i].second.y << "}";
	}
	fs << "]";

	// Store the template information
	fs << "red rules horizontal" << "[";
	for (int i = 0; i < redRuleV.size(); i++)
	{
		fs << "{:" << "x0" << redRuleV[i].first.x << "y0" << redRuleV[i].first.y << "x1" << redRuleV[i].second.x << "y1" << redRuleV[i].second.y << "}";
	}
	fs << "]";

	fs.release();
}
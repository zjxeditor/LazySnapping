#include "WatershedHelper.h"
#include "LazySnapping.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <memory>

using namespace std;
using namespace cv;

Mat Image, BackUp, PaintMask;
Point OldPt;
bool IsPressed = false;


int CurrentMode = 0;	// Indicate foreground or background, foreground as default. 0 for foreground and 1 for background.
const Scalar PaintColor[2] = { CV_RGB(0,0,255),CV_RGB(255,0,0) };
const string WindowName = "LazySnapping";
unique_ptr<WatershedHelper> WatershedProcessor;
unique_ptr<LazySnapping> LazySnappingProcessor;


void FindConnectedComponents(const Mat& mask, vector<vector<Point>>& contours, bool poly1_hull0 = true,	float perimScale = 4);
void onMouse(int event, int x, int y, int flags, void*);
void Help();
void Process();


void main()
{
	Help();

	Image = imread("images/flowers.png");
	if (Image.type() != CV_8UC3)
	{
		cout << "Input image type is not CV_8UC3" << endl;
		return;
	}
	Image.copyTo(BackUp);
	PaintMask.create(Image.size(), CV_8UC1);
	PaintMask = Scalar::all(0);

	WatershedProcessor = make_unique<WatershedHelper>(Image, 5, 5, 2, 2);
	WatershedProcessor->Process(true);
	LazySnappingProcessor = make_unique<LazySnapping>(WatershedProcessor->GetMask(), WatershedProcessor->GetColors(), WatershedProcessor->GetGraph());

	imshow(WindowName, Image);
	setMouseCallback(WindowName, onMouse, nullptr);

	while (true)
	{
		int c = cvWaitKey(0);
		c = char(c);
		if (c == 27)
		{
			// Exit.
			break;
		}
		else if (c == 'r')
		{
			BackUp.copyTo(Image);
			PaintMask = Scalar::all(0);
			CurrentMode = 0;
			imshow(WindowName, Image);
		}
		else if (c == 'b')
		{
			// Background mode.
			CurrentMode = 1;
		}
		else if (c == 'f')
		{
			// Foreground mode.
			CurrentMode = 0;
		}
		else if(c == 'k')
		{
			int temp = 64;
			cout << "Input Kmeans number: ";
			cin >> temp;
			LazySnappingProcessor->SetClusterNum(temp);
			Process();
		}
		else if(c == 'e')
		{
			float temp = 100;
			cout << "E2 weight: ";
			cin >> temp;
			LazySnappingProcessor->SetE2Weight(temp);
			Process();
		}
	}
}

void Help()
{
	cout << "Press 'f' to set foreground." << endl
		<< "Press 'b' to set background." << endl
		<< "Press 'r' to reset image." << endl
		<< "Press 'k' to set kmeans cluster number." << endl
		<< "Press 'e' to set e2 weight." << endl
		<< "--------------------------------------------------" << endl
		<< endl;
}

void Process()
{
	// Process Lazy Snapping.
	if (!LazySnappingProcessor->Process(PaintMask, true))
		return;
	Mat segmentation = LazySnappingProcessor->GetSegmentation();

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(segmentation, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	//FindConnectedComponents(segmentation, contours, true, 4);

	drawContours(Image, contours, -1, Scalar(0, 255, 0));
	imshow(WindowName, Image);
}

void onMouse(int event, int x, int y, int flags, void*)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		OldPt = Point(x, y);
		IsPressed = true;
	}
	else if (event == CV_EVENT_MOUSEMOVE && flags & CV_EVENT_FLAG_LBUTTON)
	{
		Point pt(x, y);
		line(Image, OldPt, pt, PaintColor[CurrentMode], 2);
		line(PaintMask, OldPt, pt, Scalar(CurrentMode + 1), 2);
		OldPt = pt;
		imshow(WindowName, Image);
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		if (!IsPressed)
			return;
		IsPressed = false;

		Process();
	}
}

/// <summary>
/// Find the connected components in a binary mask image.
/// </summary>
/// <param name="mask">The mask image.</param>
/// <param name="contours">The output contours.</param>
/// <param name="poly1_hull0">Set to true to use polygon approximation. Set false to use convex hull approximation.</param>
/// <param name="perimScale">The perimeter scale.</param>
void FindConnectedComponents(
	const Mat& mask,
	vector<vector<Point>>& contours,
	bool poly1_hull0,
	float perimScale)
{
	// Clean up raw mask.
	morphologyEx(mask, mask, MORPH_OPEN, NULL, Point(-1, -1), 1);
	morphologyEx(mask, mask, MORPH_CLOSE, NULL, Point(-1, -1), 1);

	// Find contours around only big regions.
	vector<vector<Point>> orgContours;
	vector<Vec4i> hierarchy;

	contours.clear();
	findContours(mask, orgContours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for each(auto& contour in orgContours)
	{
		double len = arcLength(contour, true);
		// Calculate perimeter len threshold
		double q = (mask.rows + mask.cols) / perimScale;

		// Get rid of blob if its perimeter is too small.
		if (len < q)
			continue;

		// Smooth its edge if it is large enough.
		vector<Point> res;
		if (poly1_hull0)
			// Polygonal approximation.
			approxPolyDP(contour, res, 2.0, true);
		else
			// Convex hull of the segmentation.
			convexHull(contour, res, true);
		contours.push_back(res);
	}
}
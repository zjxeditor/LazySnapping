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


void onMouse(int event, int x, int y, int flags, void*)
{
	if(event == CV_EVENT_LBUTTONDOWN)
	{
		OldPt = Point(x, y);
		IsPressed = true;
	}
	else if(event == CV_EVENT_MOUSEMOVE && flags & CV_EVENT_FLAG_LBUTTON)
	{
		Point pt(x, y);
		line(Image, OldPt, pt, PaintColor[CurrentMode], 2);
		line(PaintMask, OldPt, pt, Scalar(CurrentMode + 1), 2);
		OldPt = pt;
		imshow(WindowName, Image);
	}
	else if(event == CV_EVENT_LBUTTONUP)
	{
		if (!IsPressed)
			return;
		IsPressed = false;

		// Process Lazy Snapping.
		if (!LazySnappingProcessor->Process(PaintMask, 64, true))
			return;
		Mat segmentation = LazySnappingProcessor->GetSegmentation();

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(segmentation, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		drawContours(Image, contours, -1, Scalar(0, 255, 0));

		imshow(WindowName, Image);
	}
}


void main()
{
	Image = imread("images/carsten.jpg");
	if (Image.type() != CV_8UC3)
	{
		cout << "Input image type is not CV_8UC3" << endl;
		return;
	}
	Image.copyTo(BackUp);
	PaintMask.create(Image.size(), CV_8UC1);
	PaintMask = Scalar::all(0);

	WatershedProcessor = make_unique<WatershedHelper>(Image, 4, 4, 1, 1);
	WatershedProcessor->Process(true);
	LazySnappingProcessor = make_unique<LazySnapping>(WatershedProcessor->GetMask(), WatershedProcessor->GetColors(), WatershedProcessor->GetGraph());

	imshow(WindowName, Image);
	setMouseCallback(WindowName, onMouse, nullptr);

	while(true)
	{
		int c = cvWaitKey(0);
		c = char(c);
		if(c == 27)
		{
			// Exit.
			break;
		}
		else if(c == 'r')
		{
			BackUp.copyTo(Image);
			PaintMask = Scalar::all(0);
			CurrentMode = 0;
			imshow(WindowName, Image);
		}
		else if(c == 'b')
		{
			// Background mode.
			CurrentMode = 1;
		}
		else if(c == 'f')
		{
			// Foreground mode.
			CurrentMode = 0;
		}
	}
}
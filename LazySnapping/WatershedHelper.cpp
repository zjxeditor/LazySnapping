#include "WatershedHelper.h"
#include <opencv2/imgproc.hpp>
#include <queue>
#include <iostream>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;

// Todo: adjust seed generate parameters.
WatershedHelper::WatershedHelper(const Mat& srcImage, int hs /* = 2 */, int vs /* = 2 */, int hf /* = 2 */, int vf /* = 2 */)
	: m_compCount(0), m_hspace(hs), m_vspace(vs), m_hoffset(hf), m_voffset(vf), m_seedRowCount(0), m_seedColCount(0)
{
	// Constraint input image type.
	if (srcImage.type() != CV_8UC3)
		throw new exception("Input image type must be CV_8UC3");
	srcImage.copyTo(m_srcImage);
	
	// Create mask image.
	m_maskImage.create(m_srcImage.size(), CV_32SC1);
	m_rows = m_maskImage.rows;
	m_cols = m_maskImage.cols;
}

WatershedHelper::~WatershedHelper()
{
}

void WatershedHelper::Process(bool showRes /* = false */)
{
	generateSeeds();
	watershed(m_srcImage, m_maskImage);
	buildGraph();
	removeBorder();

	for (int i = 0; i < m_rows; i++)
	{
		int* maskptr = m_maskImage.ptr<int>(i);
		for (int j = 0; j < m_cols; j++)
		{
			if (maskptr[j] <= 0)
				cout << "Invalid pixel value." << endl;
		}
	}

	if (showRes)
		showWatershedResult();
}

void WatershedHelper::SetSrcImage(const cv::Mat& srcImage)
{
	// Constraint input image type.
	if (srcImage.type() != CV_8UC3)
		throw new exception("Input image type must be CV_8UC3");
	srcImage.copyTo(m_srcImage);

	// Create mask image.
	m_maskImage.create(m_srcImage.size(), CV_32SC1);
	m_rows = m_maskImage.rows;
	m_cols = m_maskImage.cols;
}

void WatershedHelper::SetSeedConfig(int hs, int vs, int hf, int vf)
{
	m_hspace = hs;
	m_vspace = vs;
	m_hoffset = hf;
	m_voffset = vf;
}

Mat WatershedHelper::GetMask() const
{
	Mat res;
	m_maskImage.copyTo(res);
	return res;
}
vector<Vec3b> WatershedHelper::GetColors() const { return m_nodeColors; }
vector<Connection> WatershedHelper::GetGraph() const { return m_graph; }

void WatershedHelper::buildGraph()
{
	// Use cache to mark the pixels which have been visited.
	Mat cache(m_maskImage.size(), CV_8SC1);
	cache = Scalar::all(0);
	m_nodeColors.resize(m_compCount, Vec3b(0, 0, 0));
	const Point offsets[4] = { Point(-1, 0), Point(0, -1), Point(1, 0), Point(0, 1) };

	// Border node act as the seed of one section.
	queue<Point> borderNodes;
	RNG random(getTickCount());
	while(true)
	{
		int initX = random.uniform(0, m_cols);
		int initY = random.uniform(0, m_rows);
		if(m_maskImage.at<int>(initY, initX) > 0)
		{
			borderNodes.push(Point(initX, initY));
			break;
		}
	}

	while (!borderNodes.empty())
	{
		Point startPoint = borderNodes.front();
		borderNodes.pop();
		if (cache.at<char>(startPoint) > 0)
			continue;

		queue<Point> currentNode;				// Pixel cache for current section.			
		vector<BorderElement> currentBorder;	// Border cache for current section.
		int currentComp = m_maskImage.at<int>(startPoint);
		Vec3i currentColor = { 0,0,0 };
		int currentCount = 0;
		currentNode.push(startPoint);
		cache.at<char>(startPoint) = 1;	// Mark visited.
		char visitType = 0;

		// Use BFS to traverse the current section.
		while (!currentNode.empty())
		{
			Point currentPixel = currentNode.front();
			currentNode.pop();
			currentCount++;
			currentColor += m_srcImage.at<Vec3b>(currentPixel);
			visitType = cache.at<char>(currentPixel);

			for (int i = 0; i < 4; i++)
			{
				Point adjacentPixel = currentPixel + offsets[i];
				if (!isBound(adjacentPixel) || cache.at<char>(adjacentPixel) > 0)
					continue;

				int adjacentComp = m_maskImage.at<int>(adjacentPixel);
				if (adjacentComp == currentComp)
				{
					// In the same section.
					currentNode.push(adjacentPixel);
					cache.at<char>(adjacentPixel) = 1;	// Mark visited.
					continue;
				}
				if(adjacentComp <= 0)
				{
					if (visitType == 2)	// Cannot extend along border.
						continue;
					// Meet the border. Include border pixels into current section.
					m_maskImage.at<int>(adjacentPixel) = currentComp;
					currentNode.push(adjacentPixel);
					cache.at<char>(adjacentPixel) = visitType + 1;	// Mark border.
					continue;
				}

				// Meet a new section.
				bool flag = false;
				for (size_t k = 0; k < currentBorder.size(); k++)
				{
					if (currentBorder[k].Id == adjacentComp)
					{
						// Already border, increase border length.
						flag = true;
						currentBorder[k].Length++;
						break;
					}
				}
				if (flag)
					continue;
				// Create a new element for border.
				currentBorder.push_back(BorderElement(adjacentComp, 1, adjacentPixel));
			}
		}
		
		// Get the average color.
		currentColor /= currentCount;
		m_nodeColors[currentComp - 1] = static_cast<Vec3b>(currentColor);

		// Construct graph connection relationship.
		m_graph.push_back(Connection(currentComp));
		Connection& connection = m_graph[m_graph.size() - 1];
		for each (auto& element in currentBorder)
		{
			connection.Edges.push_back(Edge(element.Id, element.Length));
			borderNodes.push(element.Pos);
		}
	}
}

void WatershedHelper::generateSeeds()
{
	if (m_hoffset >= m_srcImage.cols || m_voffset >= m_srcImage.rows)
		throw new exception("Invalid offset parameters");

	// Seed start from 1.
	m_maskImage = Scalar::all(0);
	m_compCount = 1;
	for (int i = m_voffset; i < m_rows; i += m_vspace)
	{
		for (int j = m_hoffset; j < m_cols; j += m_hspace)
		{
			m_maskImage.at<int>(i, j) = m_compCount++;
		}
	}
	m_compCount--;

	m_seedRowCount = 1 + (m_rows - m_voffset - 1) / m_vspace;
	m_seedColCount = 1 + (m_cols - m_hoffset - 1) / m_hspace;
}

void WatershedHelper::removeBorder()
{
	// Use cache to mark the pixels which have been visited.
	Mat cache(m_maskImage.size(), CV_8SC1);
	cache = Scalar::all(0);
	vector<Point> borderPosition;
	vector<int> res;
	const Point offsets[4] = { Point(0, -1), Point(-1, 0),  Point(1, 0), Point(0, 1) };
	
	for (int i = 0; i < m_rows; i++)
	{
		int* maskptr = m_maskImage.ptr<int>(i);
		for (int j = 0; j < m_cols; j++)
		{
			if (maskptr[j] <= 0)
			{
				cache = Scalar::all(0);
				Point startPoint = Point(j, i);
				queue<Point> currentNodes;
				currentNodes.push(startPoint);
				cache.at<char>(startPoint) = 1;

				borderPosition.push_back(startPoint);
				res.push_back(0);

				while (!currentNodes.empty())
				{
					Point currentPixel = currentNodes.front();
					currentNodes.pop();

					bool flag = false;
					for (int k = 0; k < 4; k++)
					{
						Point adjacentPixel = currentPixel + offsets[k];
						if (!isBound(adjacentPixel) || cache.at<char>(adjacentPixel) == 1)
							continue;

						int adjacentComp = m_maskImage.at<int>(adjacentPixel);
						if(adjacentComp <= 0)
						{
							currentNodes.push(adjacentPixel);
							cache.at<char>(adjacentPixel) = 1;	// Mark border.
							continue;
						}
						
						flag = true;
						res[res.size() - 1] = adjacentComp;
						break;
					}

					if (flag)
						break;
				}
			}
		}
	}

	for (size_t i = 0; i < borderPosition.size(); i++)
	{
		m_maskImage.at<int>(borderPosition[i]) = res[i];
	}
}

bool WatershedHelper::isBound(const Point& pos) const
{
	if (pos.x < 0 || pos.x >= m_cols || pos.y < 0 || pos.y >= m_rows)
		return false;
	return true;
}

void WatershedHelper::showWatershedResult()
{
	Mat colorRes(m_srcImage.size(), CV_8UC3);
	colorRes = Scalar::all(0);
	// Fill color.
	for (int i = 0; i < m_rows; i++)
	{
		int* maskptr = m_maskImage.ptr<int>(i);
		Vec3b* resptr = colorRes.ptr<Vec3b>(i);
		for (int j = 0; j < m_cols; j++)
		{
			if (maskptr[j] <= 0)
				resptr[j] = Vec3b(0, 0, 0);
			else
				resptr[j] = m_nodeColors[maskptr[j] - 1];
		}
	}
	imshow(WatershedWindowName, colorRes);
	// Draw graph.
	/*Point startPos, endPos;
	for each(auto& connection in m_graph)
	{
		startPos = TransCompIdToPoint(connection.Id);
		circle(colorRes, startPos, 2, Scalar(0, 255, 0));

		for each(auto& edge in connection.Edges)
		{
			endPos = TransCompIdToPoint(edge.Id);
			circle(colorRes, endPos, 2, Scalar(0, 255, 0));
			line(colorRes, startPos, endPos, Scalar(255, 0, 0));
		}
	}

	imshow(GraphWindowName, colorRes);*/
}

Point WatershedHelper::TransCompIdToPoint(int id) const
{
	int r = id / m_seedColCount;
	int c = id % m_seedColCount - 1;
	if (c < 0)
	{
		r--;
		c += m_seedColCount;
	}
	r = m_voffset + r*m_vspace;
	c = m_hoffset + c*m_hspace;

	return Point(c, r);
}



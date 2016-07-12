#include "LazySnapping.h"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

LazySnapping::LazySnapping(const cv::Mat& maskImage, const std::vector<cv::Vec3b>& nodeColors, const std::vector<Connection>& connections, int clusterNum /* = 64 */, float e2weight /* = 100.0 */)
	: m_maskImage(maskImage), m_nodeColors(nodeColors), m_connections(connections), m_clusterNum(clusterNum), m_e2weight(e2weight)
{
	if (m_maskImage.type() != CV_32SC1)
		throw new exception("Mask image type must be CV_32SC1");
	if (m_clusterNum < 1 || m_clusterNum > 100)
		throw new exception("ClusterNum must be in [1, 100].");
	if(m_e2weight <= 0)
		throw new exception("E2 weight must be a positive number.");

	m_segImage.create(m_maskImage.size(), CV_8UC1);

	int edgeCount = 0;
	for each(auto& item in m_connections)
		edgeCount += item.Edges.size();

	m_graph = make_unique<Graph<float, float, float>>(m_connections.size(), edgeCount * 2);
}

LazySnapping::~LazySnapping()
{
}


bool LazySnapping::Process(cv::Mat& paintImage, bool showSegmentation /* = false */)
{
	if (!setMarkPoints(paintImage))
		return false;
	runMaxFlow();
	BuildSegmentation();
	if (showSegmentation)
		imshow(SegWindowName, m_segImage);
	return true;
}

Mat LazySnapping::GetSegmentation() const
{
	Mat res;
	m_segImage.copyTo(res);
	return res;
}

void LazySnapping::SetClusterNum(int num)
{
	if (num < 1 || num > 100)
	{
		cout << "ClusterNum must be in [1, 100]." << endl;
		return;
	}
	m_clusterNum = num;
}

void LazySnapping::SetE2Weight(float weight)
{
	if(weight <= 0)
	{
		cout << "E2 weight must be a positive number." << endl;
		return;
	}
	m_e2weight = weight;
}
	
// Todo: change cluster number.
bool LazySnapping::setMarkPoints(cv::Mat& paintImage)
{
	if (paintImage.size() != m_maskImage.size())
		throw new exception("Image size not match.");
	if (paintImage.type() != CV_8UC1)
		throw new exception("Image type must be CV_8UC1");

	// Get foreground and background components' ids.
	m_foreComps.clear();
	m_backComps.clear();
	for (int i = 0; i < m_maskImage.rows; i++)
	{
		int* maskptr = m_maskImage.ptr<int>(i);
		uchar* paintptr = paintImage.ptr<uchar>(i);
		for (int j = 0; j < m_maskImage.cols; j++)
		{
			if(paintptr[j] == 1)		// Foreground.
				m_foreComps.push_back(maskptr[j]);
			else if(paintptr[j] == 2)	// Background.
				m_backComps.push_back(maskptr[j]);
		}
	}

	if (m_foreComps.size() == 0 || m_backComps.size() == 0)
		return false;

	// Remove redundant ids.
	sort(m_foreComps.begin(), m_foreComps.end());
	auto end_unique = unique(m_foreComps.begin(), m_foreComps.end());
	m_foreComps.erase(end_unique, m_foreComps.end());
	sort(m_backComps.begin(), m_backComps.end());
	end_unique = unique(m_backComps.begin(), m_backComps.end());
	m_backComps.erase(end_unique, m_backComps.end());

	// Use kmeans method to get cluster colors.
	// Must use float data type to do kmeans.
	vector<Vec3f> foreColors;
	vector<Vec3f> backColors;
	for each(auto& colorComp in m_foreComps)
		foreColors.push_back(m_nodeColors[colorComp - 1]);
	for each(auto& colorComp in m_backComps)
		backColors.push_back(m_nodeColors[colorComp - 1]);

	vector<int> foreLabels;
	vector<int> backLabels;
	int foreClusterNum = min(static_cast<int>(m_foreComps.size()), m_clusterNum);
	int backClusterNum = min(static_cast<int>(m_backComps.size()), m_clusterNum);
	// Todo: adjust kmeans parameters.
	kmeans(foreColors, foreClusterNum, foreLabels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS);
	kmeans(backColors, backClusterNum, backLabels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS);

	vector<Vec3i> tempForeColors(foreClusterNum, Vec3i(0, 0, 0));
	vector<Vec3i> tempBackColors(backClusterNum, Vec3i(0, 0, 0));
	m_foreColors.resize(foreClusterNum, Vec3b(0, 0, 0));
	m_backColors.resize(backClusterNum, Vec3b(0, 0, 0));
	vector<int> counter(foreClusterNum, 0);
	for (size_t i = 0; i < foreLabels.size(); i++)
	{
		tempForeColors[foreLabels[i]] += foreColors[i];
		counter[foreLabels[i]]++;
	}
	for(int i = 0; i < foreClusterNum; i++)
	{
		if (counter[i] == 0)
			throw new exception("Cluster has no element.");
		m_foreColors[i] = tempForeColors[i] / counter[i];
	}
	counter.resize(backClusterNum, 0);
	for (size_t i = 0; i < backLabels.size(); i++)
	{
		tempBackColors[backLabels[i]] += backColors[i];
		counter[backLabels[i]]++;
	}
	for (int i = 0; i < backClusterNum; i++)
	{
		if (counter[i] == 0)
			throw new exception("Cluster has no element.");
		m_backColors[i] = tempBackColors[i] / counter[i];
	}

	return true;
}

// Todo: analyze max flow graph build process.
void LazySnapping::runMaxFlow()
{
	// Add nodes.
	for (size_t i = 0; i < m_nodeColors.size(); i++)
	{
		m_graph->add_node();
		Point2f e1 = calE1(i + 1);
		m_graph->add_tweights(i, e1.x, e1.y);
	}
	// Add edges.
	int startId, endId;
	for each(auto& connection in m_connections)
	{
		startId = connection.Id;
		for each(auto& item in connection.Edges)
		{
			endId = item.Id;
			float e2 = calE2(startId, endId);
			m_graph->add_edge(startId - 1, endId - 1, e2, e2);
		}
	}

	m_graph->maxflow();
}

void LazySnapping::BuildSegmentation()
{
	m_segImage = Scalar::all(0);
	for (int i = 0; i < m_segImage.rows; i++)
	{
		int* maskptr = m_maskImage.ptr<int>(i);
		uchar* segptr = m_segImage.ptr<uchar>(i);
		for (int j = 0; j < m_segImage.cols; j++)
		{
			if (m_graph->what_segment(maskptr[j] - 1) == Graph<float, float, float>::SINK)
				segptr[j] = 255;
		}
	}
}

Point2f LazySnapping::calE1(int compId)
{
	if (compId < 1 || compId > static_cast<int>(m_nodeColors.size()))
		throw new exception("No such component id.");

	for each(auto& comp in m_foreComps)
	{
		// In the foreground.
		if (compId == comp)
			return Point2f(0, Infinite);
	}
	for each(auto& comp in m_backComps)
	{
		// In the background.
		if (compId == comp)
			return Point2f(Infinite, 0);
	}

	Vec3b currentColor = m_nodeColors[compId - 1];
	float df = minDistance(currentColor, m_foreColors);
	float db = minDistance(currentColor, m_backColors);
	return Point2f(df / (df + db), db / (df + db));
}

// Todo: adjust E2 calculation method.
float LazySnapping::calE2(int compA, int compB)
{
	Vec3b colorA = m_nodeColors[compA - 1];
	Vec3b colorB = m_nodeColors[compB - 1];

	Vec3i diff = static_cast<Vec3i>(colorA) - static_cast<Vec3i>(colorB);
	int distance = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
	float epsilon = 0.01f;	// 0.01

	return m_e2weight / (epsilon + distance);
}

int LazySnapping::transPointToCompId(const Point& pos)
{
	if (pos.x < 0 || pos.x >= m_maskImage.cols || pos.y < 0 || pos.y >= m_maskImage.rows)
		throw new exception("Point out out of image bound.");

	return m_maskImage.at<int>(pos);
}

float LazySnapping::colorDistance(const Vec3b& colorA, const Vec3b& colorB) const
{
	Vec3i diff = static_cast<Vec3i>(colorA) - static_cast<Vec3i>(colorB);
	return static_cast<float>(sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]));
}

float LazySnapping::minDistance(const cv::Vec3b& color, const std::vector<cv::Vec3b>& collection) const
{
	float res = Infinite;
	float temp = 0;
	for each(auto& element in collection)
	{
		temp = colorDistance(color, element);
		if (temp < res)
			res = temp;
	}
	return res;
}

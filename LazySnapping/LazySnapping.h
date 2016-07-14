#pragma once

#include<opencv2/core.hpp>
#include <vector>
#include <memory>
#include <string>
#include "WatershedHelper.h"
#include "graph.h"

/// <summary>
/// Use lazy snapping algorithm to do image cut.
/// </summary>
class LazySnapping
{
public:
	LazySnapping(const cv::Mat& maskImage, const std::vector<cv::Vec3b>& nodeColors, const std::vector<Connection>& connections, int clusterNum = 64, float e2weight = 1000.0);
	~LazySnapping();

public:
	/// <summary>
	/// Do lazy snapping base on foreground points and background points.
	/// </summary>
	/// <param name="paintImage">The paint image. 1 for foreground mark, 2 for background mark.</param>
	/// <param name="showSegmentation">Set to true to show the final segmentation result.</param>
	/// <returns>True for successful operation.</returns>
	bool Process(cv::Mat& paintImage, bool showSegmentation = false);

	/// <summary>
	/// Get the final segmentation image. 255 for foreground and 0 for background.
	/// </summary>
	/// <returns></returns>
	cv::Mat GetSegmentation() const;

	/// <summary>
	/// Set kmeans cluster number.
	/// </summary>
	void SetClusterNum(int num);

	/// <summary>
	/// Set prior energy weight relative to likelihood energy.
	/// </summary>
	void SetE2Weight(float weight);

private:
	/// <summary>
	/// Set the foreground and background mark points.
	/// </summary>
	/// <param name="paintImage">The paint image. 1 for foreground mark, 2 for background mark.</param>
	/// <returns>True for successful operation.</returns>
	bool setMarkPoints(cv::Mat& paintImage);

	/// <summary>
	/// Run the maximum flow algorithm.
	/// </summary>
	void runMaxFlow();

	/// <summary>
	/// Build the segmentation image.
	/// </summary>
	void BuildSegmentation();

	/// <summary>
	/// Calculate the likelihood energy specific component.
	/// In the result, x stores the foreground energy and y stores the background energy.
	/// </summary>
	cv::Point2f calE1(int compId);

	/// <summary>
	/// Calculate prior energy between two components.
	/// </summary>
	float calE2(int compA, int compB);

	/// <summary>
	/// Transform the point position to component id according to the mask image.
	/// </summary>
	int transPointToCompId(const cv::Point& pos);

	/// <summary>
	/// Calculate the Euclid distance.
	/// </summary>
	/// <returns></returns>
	float colorDistance(const cv::Vec3b& colorA, const cv::Vec3b& colorB) const;

	/// <summary>
	/// Calculate the minimum distance to a color collection.
	/// </summary>
	/// <param name="color">The color value.</param>
	/// <param name="collection">The color collection.</param>
	float minDistance(const cv::Vec3b& color, const std::vector<cv::Vec3b>& collection) const;

private:
	std::vector<int> m_foreComps;
	std::vector<int> m_backComps;
	std::vector<cv::Vec3b> m_foreColors;
	std::vector<cv::Vec3b> m_backColors;

	cv::Mat m_maskImage;
	std::vector<cv::Vec3b> m_nodeColors;
	std::vector<Connection> m_connections;	
	std::unique_ptr<Graph<float, float, float>> m_graph;

	cv::Mat m_segImage;
	const float Infinite = 1e10;
	const std::string SegWindowName = "Segmentation";

	int m_clusterNum;
	float m_e2weight;
};


#pragma once

#include<opencv2/core.hpp>
#include <vector>
#include <string>

/// <summary>
/// One section edge. It record the adjacent section's id and border length.
/// </summary>
struct Edge
{
	Edge() {}
	Edge(int id, int length) : Id(id), Length(length) {}

	int Id = 0;
	int Length = 0;
};

/// <summary>
/// Node relationship in graph.
/// </summary>
struct Connection
{
	Connection() {}
	Connection(int id) : Id(id)	{}

	int Id = 0;
	std::vector<Edge> Edges;
};

/// <summary>
/// Section border information. It record adjacent section's id, border length and seed.
/// </summary>
struct BorderElement
{
	BorderElement() {}
	BorderElement(int id, int length, cv::Point pos) : Id(id), Length(length), Pos(pos)	{}

	int Id = 0;
	int Length = 0;
	cv::Point Pos;
};


/// <summary>
/// Segment image using watershed algorithm to generate super pixels.
/// </summary>
class WatershedHelper
{
public:
	WatershedHelper(const cv::Mat& srcImage, int hs = 2, int vs = 2, int hf = 2, int vf = 2);
	~WatershedHelper();

public:
	/// <summary>
	/// Watershed the source image. Each watershed section is filled with the average color.
	/// The directed graph will be build.
	/// </summary>
	/// <param name="showWatershed">Set to true to show watershed result.</param>
	void Process(bool showWatershed = false);

	/// <summary>
	/// Set the source image.
	/// </summary>
	/// <param name="srcImage">The source image.</param>
	void SetSrcImage(const cv::Mat& srcImage);

	/// <summary>
	/// Sets the seed configuration for "generateSeeds" method.
	/// </summary>
	/// <param name="hs">The horizon space.</param>
	/// <param name="vs">The vertical space.</param>
	/// <param name="hf">The horizon offset.</param>
	/// <param name="vf">The vertical offset.</param>
	void SetSeedConfig(int hs, int vs, int hf, int vf);

	cv::Mat GetMask() const;
	std::vector<cv::Vec3b> GetColors() const;
	std::vector<Connection> GetGraph() const;

private:
	/// <summary>
	/// Generate seed points uniformly.
	/// </summary>
	void generateSeeds();

	/// <summary>
	/// Remove the watershed border residue after build graph process. Use BFS method. 
	/// </summary>
	void removeBorder();

	/// <summary>
	/// Build the watershed graph. It will be a directed graph.
	/// </summary>
	void buildGraph();

	/// <summary>
	/// Determine whether the specified point is in the image area.
	/// </summary>
	/// <param name="pos">The test point position.</param>
	/// <returns>True for bounded.</returns>
	bool isBound(const cv::Point& pos) const;

	/// <summary>
	/// Show the watershed process result. Every section is filled with average color
	/// and the built graph.
	/// </summary>
	void showWatershedResult();

	/// <summary>
	/// Transform component id to the seed position on image. Used when 
	/// showing the watered result.
	/// </summary>
	/// <param name="id">The componet id value.</param>
	/// <returns></returns>
	cv::Point TransCompIdToPoint(int id) const;

private:
	cv::Mat m_srcImage;
	cv::Mat m_maskImage;

	std::vector<cv::Vec3b> m_nodeColors;
	std::vector<Connection> m_graph;

	int m_compCount;

	int m_hspace;
	int m_vspace;
	int m_hoffset;
	int m_voffset;

	int m_rows;
	int m_cols;
	int m_seedRowCount;
	int m_seedColCount;

	const std::string WatershedWindowName = "Watershed";
	const std::string GraphWindowName = "Graph";
};

	
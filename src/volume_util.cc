#include "volume_util.h"

#include <fstream>
#include <iomanip>

#include "file_manager.h"

using namespace cv;
using namespace filesystem;

cv::Mat volread(const cv::String &filename, int flags) {
	std::vector<cv::Mat> planes;
	path vol_path(filename);
	path files_path = vol_path / path("files.txt");		
	std::vector<std::string> planenames;

	{
		std::ifstream is(files_path.string());				// text mode shoud translate \r\n into \n only 
		if (!is) {
			return Mat();
		}

		std::string cur_filename;
		while (std::getline(is, cur_filename)) {
			planenames.push_back(cur_filename);
		}
	}
		
	int sz[3];
	int type;

	for (unsigned int plane = 0; plane < planenames.size(); plane++) {
		Mat tmp = imread((vol_path / path(planenames[plane])).string(), flags);
		if (tmp.empty()) {
			return Mat();
		}
		if (plane == 0) {
			// We can set volume sizes and type when we see the first plane
			sz[0] = static_cast<int>(planenames.size());
			sz[1] = tmp.rows;
			sz[2] = tmp.cols;
			type = tmp.type();
		}
		else {
			if (tmp.rows != sz[1] || tmp.cols != sz[2]) {
				return Mat();
			}
		}
		planes.push_back(std::move(tmp));
	}

	Mat volume(3, sz, type);
	uchar *plane_data = volume.data;

	// Every matrix in the array is guaranteed to be continuous because it was created with Mat::create()
	for (Mat& plane : planes) {
		if (!plane.isContinuous())
			return Mat();
		size_t plane_size = plane.size[0] * plane.step[0];
		memcpy(plane_data, plane.data, plane_size);
		plane_data += plane_size;
	}
	return volume;
}

bool volwrite(const cv::String& filename, const cv::Mat& volume) {
	if (volume.empty() || volume.dims != 3)
		return false;

	int rows = volume.size[1];
	int cols = volume.size[2];

	std::vector<Mat> planes;

	size_t step = volume.step[1];

	for (int plane = 0; plane < volume.size[0]; plane++) {
		planes.emplace_back(rows, cols, volume.type(), volume.data + volume.step[0] * plane, step);
	}

	path vol_path(filename);

	if (!create_directories(vol_path))
		return false;

	path files_path = vol_path / path("files.txt");

	std::ofstream os(files_path.string(), std::ios::binary);
	if (!os)
		return false;

	for (unsigned i = 0; i < planes.size(); i++) {
		std::ostringstream plane_name;
		plane_name << std::setw(4) << std::setfill('0') << (i + 1) << ".png";						// this should be made general
		if (!imwrite((vol_path / path(plane_name.str())).string(), planes[i]))
			return false;
		os << plane_name.str() << '\n';
	}
	return true;
}
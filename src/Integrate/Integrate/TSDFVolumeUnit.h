#pragma once
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <pcl/common/time.h>

struct FramedTransformation {
	int frame1_;
	int frame2_;
	Eigen::Matrix4d transformation_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	FramedTransformation(int frame1, int frame2, const Eigen::Matrix4d& t)
		: frame1_(frame1), frame2_(frame2), transformation_(t)
	{}
};



struct RGBDTrajectory {
	std::vector< FramedTransformation, Eigen::aligned_allocator<FramedTransformation>> data_;

	void LoadFromFile(std::string filename) {
		data_.clear();
		int frame1, frame2;
		Eigen::Matrix4d trans;
		FILE * f = fopen(filename.c_str(), "r");
		if (f != NULL) {
			char buffer[1024];
			while (fgets(buffer, 1024, f) != NULL) {
				if (strlen(buffer) > 0 && buffer[0] != '#') {
					sscanf(buffer, "%d %d", &frame1, &frame2);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf %lf", &trans(0, 0), &trans(0, 1), &trans(0, 2), &trans(0, 3));
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf %lf", &trans(1, 0), &trans(1, 1), &trans(1, 2), &trans(1, 3));
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf %lf", &trans(2, 0), &trans(2, 1), &trans(2, 2), &trans(2, 3));
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf %lf", &trans(3, 0), &trans(3, 1), &trans(3, 2), &trans(3, 3));

					data_.push_back(FramedTransformation(frame1, frame2, trans));
				}
			}
			fclose(f);
		}
	}
	void SaveToFile(std::string filename) {
		FILE * f = fopen(filename.c_str(), "w");
		for (int i = 0; i < (int)data_.size(); i++) {
			Eigen::Matrix4d & trans = data_[i].transformation_;
			fprintf(f, "%d\t%d\n", data_[i].frame1_, data_[i].frame2_);
			fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(0, 0), trans(0, 1), trans(0, 2), trans(0, 3));
			fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(1, 0), trans(1, 1), trans(1, 2), trans(1, 3));
			fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(2, 0), trans(2, 1), trans(2, 2), trans(2, 3));
			fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(3, 0), trans(3, 1), trans(3, 2), trans(3, 3));
		}
		fclose(f);
	}

	void SaveToFileAppend(std::string filename)
	{
		FILE * f = fopen(filename.c_str(), "a+");
		for (int i = 0; i < (int)data_.size(); i++) {
			Eigen::Matrix4d & trans = data_[i].transformation_;
			fprintf(f, "%d\t%d\n", data_[i].frame1_, data_[i].frame2_);
			fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(0, 0), trans(0, 1), trans(0, 2), trans(0, 3));
			fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(1, 0), trans(1, 1), trans(1, 2), trans(1, 3));
			fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(2, 0), trans(2, 1), trans(2, 2), trans(2, 3));
			fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(3, 0), trans(3, 1), trans(3, 2), trans(3, 3));
		}
		fclose(f);
	}
};



struct CameraParam {
public:
	double fx_, fy_, cx_, cy_;
	double k1_, k2_, k3_, p1_, p2_;
	double depth_ratio_;
	double downsample_leaf;

	int img_width_;
	int img_height_;

	double integration_trunc_;
	
	CameraParam() :
		fx_(525.0f), fy_(525.0f), cx_(319.5f), cy_(239.5f),
		k1_(0.0), k2_(0.0), k3_(0.0), p1_(0.0), p2_(0.0),
		depth_ratio_(1000.0),
		downsample_leaf(0.05),
		img_width_(640), img_height_(480),
		integration_trunc_(4.0)
	{}

	void LoadFromFile(std::string filename) {
		FILE * f = fopen(filename.c_str(), "r");
		if (f != NULL) {
			char buffer[1024];
			while (fgets(buffer, 1024, f) != NULL) {
				if (strlen(buffer) > 0 && buffer[0] != '#') {
					sscanf(buffer, "%lf", &fx_);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf", &fy_);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf", &cx_);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf", &cy_);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf", &k1_);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf", &k2_);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf", &k3_);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf", &p1_);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf", &p2_);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf", &depth_ratio_);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf", &downsample_leaf);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%d", &img_width_);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%d", &img_height_);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf", &integration_trunc_);
				}
			}
			printf("Camera model set to (fx: %.2f, fy: %.2f, cx: %.2f, cy: %.2f, k1: %.2f, k2: %.2f, k3: %.2f, p1: %.2f, p2: %.2f, depth_ratio_: %.2f, downsample_leaf: %.2f, img_width_: %d, img_height_: %d, integration_trunc: %.2f)\n",
				fx_, fy_, cx_, cy_, k1_, k2_, k3_, p1_, p2_, depth_ratio_, downsample_leaf, img_width_, img_height_, integration_trunc_);
			fclose(f);
		}
	}
};

class TSDFVolumeUnit
{
public:
	TSDFVolumeUnit( int resolution, int xi, int yi, int zi );
	~TSDFVolumeUnit(void);

public:
	typedef boost::shared_ptr< TSDFVolumeUnit > Ptr;

public:
	float * sdf_;
	float * weight_;

	const int resolution_;
	int xi_, yi_, zi_;
};


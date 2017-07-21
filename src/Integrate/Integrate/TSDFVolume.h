#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <unordered_set>
#include <unordered_map>

#include "TSDFVolumeUnit.h"

/////////////////////////////////////////////////////
// hash function of TSDFVolumeUnit
// -256 ~ 256 on each axis
// indexing of the volumes
// x * 64 ~ x * 64 + 63
/////////////////////////////////////////////////////

class TSDFVolume
{
public:
	TSDFVolume( int cols, int rows, std::string camera_file );
	~TSDFVolume(void);

public:

	CameraParam camera_;
	int cols_, rows_;

	double unit_length_;
	double tsdf_trunc_;
	double integration_trunc_;

	std::unordered_map< int, TSDFVolumeUnit::Ptr > data_;

public:
	void ScaleDepth( std::vector< unsigned short > & depth, std::vector< float > & scaled );
	void Integrate( std::vector< unsigned short > & depth, std::vector< float > & scaled, Eigen::Matrix4d & transformation );
	void IntegrateVolumeUnit( std::vector< float > & scaled, const Eigen::Matrix4f & trans, const Eigen::Matrix4f & trans_inv, TSDFVolumeUnit::Ptr volume, float x_shift, float y_shift, float z_shift );
	void SaveWorld( std::string filename );


public:
	int round( double x ) {
		return static_cast< int >( floor( x + 0.5 ) );
	}

	bool UVD2XYZ(int u, int v, unsigned short d, double & x, double & y, double & z) {
		if (d > 0) {
			cv::Mat cam(3, 3, CV_32F);
			cam.at<float>(0, 0) = camera_.fx_; cam.at<float>(0, 1) = 0; cam.at<float>(0, 2) = camera_.cx_;
			cam.at<float>(1, 0) = 0; cam.at<float>(1, 1) = camera_.fy_; cam.at<float>(1, 2) = camera_.cy_;
			cam.at<float>(2, 0) = 0; cam.at<float>(2, 1) = 0; cam.at<float>(2, 2) = 1;
			cv::Mat dist_coef(1, 5, CV_32F);
			dist_coef.at<float>(0, 0) = camera_.k1_;
			dist_coef.at<float>(0, 1) = camera_.k2_;
			dist_coef.at<float>(0, 2) = camera_.p1_;
			dist_coef.at<float>(0, 3) = camera_.p2_;
			dist_coef.at<float>(0, 4) = camera_.k3_;

			cv::Mat mat(1, 2, CV_32F);
			mat.at<float>(0, 0) = u;
			mat.at<float>(0, 1) = v;
			mat = mat.reshape(2);
			cv::undistortPoints(mat, mat, cam, dist_coef, cv::Mat(), cam);
			mat = mat.reshape(1);

			float uu = mat.at<float>(0, 0);
			float vv = mat.at<float>(0, 1);

			z = d / camera_.depth_ratio_;
			x = (uu - camera_.cx_) * z / camera_.fx_;
			y = (vv - camera_.cy_) * z / camera_.fy_;


			// ideal model
			/*z = d / camera_.depth_ratio_;
			x = (u - camera_.cx_) * z / camera_.fx_;
			y = (v - camera_.cy_) * z / camera_.fy_;*/
			return true;
		}
		else {
			return false;
		}
	}

	bool XYZ2UVD(double x, double y, double z, int & u, int & v, unsigned short & d) {
		if (z > 0) {
			float x_ = x / z;	float y_ = y / z;
			float r2 = x_*x_ + y_*y_;
			float x__ = x_*(1 + camera_.k1_*r2 + camera_.k2_*r2*r2 + camera_.k3_*r2*r2*r2) + 2 * camera_.p1_*x_*y_ + camera_.p2_*(r2 + 2 * x_*x_);
			float y__ = y_*(1 + camera_.k1_*r2 + camera_.k2_*r2*r2 + camera_.k3_*r2*r2*r2) + camera_.p1_*(r2 + 2 * y_*y_) + 2 * camera_.p2_*x_*y_;
			float uu = x__ * camera_.fx_ + camera_.cx_;
			float vv = y__ * camera_.fy_ + camera_.cy_;
			u = round(uu);
			v = round(vv);
			d = static_cast<unsigned short>(round(z * camera_.depth_ratio_));

			// ideal model
			/*u = round(x * camera_.fx_ / z + camera_.cx_);
			v = round(y * camera_.fy_ / z + camera_.cy_);
			d = static_cast<unsigned short>(round(z * camera_.depth_ratio_));*/
			return (u >= 0 && u < camera_.img_width_ && v >= 0 && v < camera_.img_height_);
		}
		else {
			return false;
		}
	}

	int hash_key( int x, int y, int z ) {
		return x * 512 * 512 + y * 512 + z;
	}

	float I2F( int i ) {
		return ( ( i - 256 ) * 64 * unit_length_ );
	}
};


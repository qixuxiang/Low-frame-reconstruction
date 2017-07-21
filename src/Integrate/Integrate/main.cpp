#include <iostream>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <boost/filesystem.hpp>

#include "TSDFVolume.h"


int main(int argc, char** argv)
{
	int width, height;
	std::string camera_file, pose_file, pointcloud_dir, world;
	if (argc != 7)
	{
		std::cout << "Integrate.exe img_width[num] img_height[num] camera_file[input_file] pose_file[input_file] pointcloud_dir[input_dir] world_pcd[output_file]\n";
		return -1;
	}	

	width = atoi(argv[1]);
	height = atoi(argv[2]);
	camera_file = argv[3];
	pose_file = argv[4];
	pointcloud_dir = argv[5];
	world = argv[6];

	/*camera_file = "C:/Users/range/Desktop/reconst/sandbox/camera_para.txt";
	pose_file = "C:/Users/range/Desktop/reconst/sandbox/pose_opt2.txt";
	pointcloud_dir = "C:/Users/range/Desktop/reconst/sandbox/pointcloud_xyzn/";

	world = "C:/Users/range/Desktop/reconst/sandbox/world.pcd";*/

	if (!boost::filesystem::exists(pose_file))
	{
		std::cout << "Must input camera pose file\n";
		return -1;
	}

	// load camera pose
	RGBDTrajectory camera_pose;
	camera_pose.LoadFromFile(pose_file);

	int num_of_pc = std::count_if(
		boost::filesystem::directory_iterator(boost::filesystem::path(pointcloud_dir)),
		boost::filesystem::directory_iterator(),
		[](const boost::filesystem::directory_entry& e) {
		return e.path().extension() == ".pcd";  });

	int img_width = width;
	int img_height = height;

	TSDFVolume tsdf(img_width, img_height, camera_file);

	for (int i = 0; i < num_of_pc;++i)
	{
		std::cout << "Integrate frame" << i << "...";
		// load scene
		std::stringstream pcss1;
		pcss1 << pointcloud_dir << "pointcloud_xyzn" << i << ".pcd";

		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		pcl::io::loadPCDFile(pcss1.str(), *scene);

		// convert to depth array
		std::vector< unsigned short > depth(img_width*img_height, 0);
		for (int t = 0; t < scene->size(); ++t)
		{
			double x, y, z;
			x = scene->points[t].x;		y = scene->points[t].y;		z = scene->points[t].z;

			int u = -1, v = -1;
			unsigned short d;
			tsdf.XYZ2UVD(x, y, z, u, v, d);
			assert(u != -1 && v != -1);

			depth[v*img_width + u] = d;
		}

		// camera pose
		Eigen::Matrix4d cp = camera_pose.data_[i].transformation_;

		// integrate
		std::vector< float > scaled(img_width*img_height, 0.0);
		tsdf.ScaleDepth(depth, scaled);
		tsdf.Integrate(depth, scaled, cp);

		std::cout << "Done!\n";
	}

	tsdf.SaveWorld(world);

	return 0;
}

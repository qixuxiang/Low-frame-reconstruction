#include <vector>
#include <iostream>
#include <string>
#include <set>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/correspondence.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp> 

#include <boost/filesystem.hpp>

#include "SubsidiaryFunction.h"
#include "GraphMatching.h"
#include "BuildCorpPointSet.h"

//static int pick_num = 0;
//static void point_picking_callback(const pcl::visualization::PointPickingEvent& event, void* viewer_void)
//{
//	std::cout << "Picking event active" << std::endl;
//	if (event.getPointIndex() != -1)
//	{
//		float x, y, z;
//		event.getPoint(x, y, z);
//		int index = event.getPointIndex();
//		std::cout << "you select point " << index << " at " << x << "," << y << "," << z << std::endl;
//		pcl::PointXYZ p;
//		p.x = x;	p.y = y;	p.z = z;
//		pcl::PointCloud<pcl::PointXYZ>::Ptr select_point(new pcl::PointCloud<pcl::PointXYZ>);
//		select_point->points.push_back(p);
//
//		boost::shared_ptr<pcl::visualization::PCLVisualizer>* viewer_ptr = static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer>*> (viewer_void);
//		boost::shared_ptr<pcl::visualization::PCLVisualizer>& viewer = *viewer_ptr;
//
//		std::stringstream ss;
//		ss << "selected_point" << pick_num;
//		std::string str_id = ss.str();
//		viewer->addPointCloud(select_point, str_id.c_str());
//		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, str_id.c_str());
//		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, str_id.c_str());
//
//		pick_num++;
//	}
//}


int main(int argc, char** argv)
{
	double score_max_depth;
	std::string pointcloud_dir, pointcloud_ds_dir, depth_dir, correspendence_file, traj_file, info_file, camera_file;

	if (argc != 9)
	{
		std::cout << "Usage:\n\ColorCorrespondence.exe pointcloud_dir[input_dir] pointcloud_ds_dir[input_dir] depth_dir[input_dir] correspendence_file[input_file] camera_file[input_file] score_max_depth[<0: will ignore]"
			<< " traj_file[output_file] info_file[output_file]\n";
		return 0;
	}

	pointcloud_dir = argv[1];
	pointcloud_ds_dir = argv[2];
	depth_dir = argv[3];
	correspendence_file = argv[4];
	camera_file = argv[5];
	score_max_depth = atof(argv[6]);

	traj_file = argv[7];
	info_file = argv[8];
	
	/*score_max_depth = -4.0;
	pointcloud_dir = "E:/Rec/ICL_NUIM_living_room/sandbox_2/pointcloud/";
	pointcloud_ds_dir = "E:/Rec/ICL_NUIM_living_room/sandbox_2/pointcloud_ds/";
	depth_dir = "E:/Rec/ICL_NUIM_living_room/sandbox_2/depth/";
	camera_file = "E:/Rec/ICL_NUIM_living_room/sandbox_2/camera_para.txt";

	std::string sift_correspendence_file = "E:/Rec/ICL_NUIM_living_room/sandbox_2/sift_correspondence.txt";
	std::string corner_correspendence_file = "E:/Rec/ICL_NUIM_living_room/sandbox_2/corner_correspondence.txt";

	std::string sift_traj_file = "C:/Users/range/Desktop/sift_traj.txt";
	std::string sift_info_file = "C:/Users/range/Desktop/sift_info.txt";
	std::string corner_traj_file = "C:/Users/range/Desktop/corner_traj.txt";
	std::string corner_info_file = "C:/Users/range/Desktop/corner_info.txt";

	int nu;
	std::cout << "Input 1 will build correspondence with SIFT, 2 will build correspondence with Shi-Tomasi\n";
	std::cin >> nu;
	if (nu == 1)
	{
	correspendence_file = sift_correspendence_file;
	traj_file = sift_traj_file;
	info_file = sift_info_file;
	}
	else if (nu == 2)
	{
	correspendence_file = corner_correspendence_file;
	traj_file = corner_traj_file;
	info_file = corner_info_file;
	}
	else
	{
	std::cout << "Input error\n";
	return -1;
	}*/

	int num_of_pc = std::count_if(
		boost::filesystem::directory_iterator(boost::filesystem::path(depth_dir)),
		boost::filesystem::directory_iterator(),
		[](const boost::filesystem::directory_entry& e) {
		return e.path().extension() == ".png";  });

	// CameraParam
	CameraParam camera;
	camera.LoadFromFile(camera_file);

	CorrespondencePixel cor_pixel;
	cor_pixel.LoadFromFile(correspendence_file);

	// Load all downsample pointcloud
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> downsample_pc;
	for (int i = 0; i < num_of_pc;++i)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_ds(new pcl::PointCloud<pcl::PointXYZRGB>);

		std::stringstream dsss1;
		dsss1 << pointcloud_ds_dir << "pointcloud_ds" << i << ".pcd";
		pcl::io::loadPCDFile(dsss1.str(), *scene_ds);
		downsample_pc.push_back(scene_ds);
	}

	// Load all depth images
	std::vector<cv::Mat> depth_img;
	for (int i = 0; i < num_of_pc;++i)
	{
		std::stringstream ss1;
		ss1 << depth_dir << "depth" << i << ".png";
		cv::Mat depth_image = cv::imread(ss1.str(), CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
		depth_image.convertTo(depth_image, CV_16U);
		depth_img.push_back(depth_image);
	}

	RGBDInformation info;	info.data_.resize(num_of_pc*num_of_pc);
	RGBDTrajectory  traj;	traj.data_.resize(num_of_pc*num_of_pc);

	/*int id1, id2;
	std::cout << "Input you want to see two image after apply pose(SIFT)\n";
	std::cin >> id1 >> id2;*/

	if (boost::filesystem::exists(traj_file))
		boost::filesystem::remove(traj_file);
	if (boost::filesystem::exists(info_file))
		boost::filesystem::remove(info_file);

#pragma omp parallel for num_threads( 8 ) schedule( dynamic )
	for (int i = 0; i < cor_pixel.data_.size(); ++i)
	{
		int img1 = cor_pixel.data_[i].imageid1_;
		int img2 = cor_pixel.data_[i].imageid2_;

		/*if (img1 != id1 || img2 != id2)
			continue;*/

		std::cout << "Begin registration correspondence between (" << img1 << ", " << img2 << ")\n";

		cv::Mat& depth_image1 = depth_img[img1];
		cv::Mat& depth_image2 = depth_img[img2];

		// prepare
		pcl::CorrespondencesPtr corps(new pcl::Correspondences);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_keypoints1(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_keypoints2(new pcl::PointCloud<pcl::PointXYZRGB>);

		for (int t = 0; t < cor_pixel.data_[i].pixel_correspondence_.size(); ++t)
		{
			int image1_u = cor_pixel.data_[i].pixel_correspondence_[t].image1_pixel_x;
			int image1_v = cor_pixel.data_[i].pixel_correspondence_[t].image1_pixel_y;
			pcl::PointXYZRGB p1 = SearchNearestValidPoint(image1_u, image1_v, depth_image1, camera);

			int image2_u = cor_pixel.data_[i].pixel_correspondence_[t].image2_pixel_x;
			int image2_v = cor_pixel.data_[i].pixel_correspondence_[t].image2_pixel_y;
			pcl::PointXYZRGB p2 = SearchNearestValidPoint(image2_u, image2_v, depth_image2, camera);

			if (p1.z < 0.0 || p2.z < 0.0)
				continue;

			pointcloud_keypoints1->push_back(p1);
			pointcloud_keypoints2->push_back(p2);

			pcl::Correspondence c;
			c.index_query = pointcloud_keypoints1->size() - 1;	c.index_match = pointcloud_keypoints2->size() - 1;
			corps->push_back(c);
		}
		if (corps->size() < 4)
			continue;

		//Graph matching
		std::cout << "Graph matching...";
		GraphMatching gm(*pointcloud_keypoints1, *pointcloud_keypoints2, *corps);
		pcl::CorrespondencesPtr	graph_corps = gm.ComputeCorrespondenceByEigenVec();
		std::cout << "Done!\n";

		if (graph_corps->size() < 4)
		{
			std::cout << "cannot find enough correspondence\n";
			continue;
		}

		Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();

		// result
		transformation_matrix = transformation_matrix*gm._transformation;

		//////////////////////////
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene1_ds = downsample_pc[img1];
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene2_ds = downsample_pc[img2];

		BuildCorpPointSet buidCorp;
		int K_set = buidCorp.ComputeCorrepondencePointSet(transformation_matrix, scene1_ds, scene2_ds, score_max_depth);

		if (K_set < buidCorp._total_size / 3 || K_set == 0 || buidCorp._total_size == 0)
		{
			std::cout << "Align fail!\n\n\n";
			continue;
		}

		Eigen::Matrix< double, 6, 6 > info_mat = buidCorp.ComputeInfoMatrix(buidCorp._correspondences, scene1_ds, scene2_ds);

		std::cout << K_set << " point pair are found as correspondence in point cloud. " << std::endl;
		std::cout << "Align successfully!\n\n";

		double score = (double)K_set / (double)buidCorp._total_size;

		InformationMatrix im = info_mat;
		FramedInformation fi(img1, img2, im, score);
		info.data_[img1*num_of_pc + img2] = fi;

		Eigen::Matrix4d tf = transformation_matrix.cast<double>();
		FramedTransformation ft(img1, img2, tf);
		traj.data_[img1*num_of_pc + img2] = ft;

		//-----------from here, below code for showing--------------------- 

		//pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene1(new pcl::PointCloud<pcl::PointXYZRGB>);
		//std::stringstream pcss1;
		//pcss1 << pointcloud_dir << "pointcloud" << img1 << ".pcd";
		//std::cout << "Load scene" << img1 << "...";
		//pcl::io::loadPCDFile(pcss1.str(), *scene1);			// WARNING: multiple thread unsafe 
		//std::cout << "Done!\n";

		//pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene2(new pcl::PointCloud<pcl::PointXYZRGB>);
		//std::stringstream pcss2;
		//pcss2 << pointcloud_dir << "pointcloud" << img2 << ".pcd";
		//std::cout << "Load scene" << img2 << "...";
		//pcl::io::loadPCDFile(pcss2.str(), *scene2);
		//std::cout << "Done!\n";

		//---------------------show correspondence------------------------------
		/*pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame2(new pcl::PointCloud<pcl::PointXYZRGB>);
		for (int t = 0; t < scene2->points.size(); ++t)
		{
			pcl::PointXYZRGB p = scene2->points[t];
			p.x += 3;
			p.z += 1;
			frame2->points.push_back(p);
		}

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame2_kp(new pcl::PointCloud<pcl::PointXYZRGB>);
		for (int t = 0; t < pointcloud_keypoints2->points.size(); ++t)
		{
			pcl::PointXYZRGB p = pointcloud_keypoints2->points[t];
			p.x += 3;
			p.z += 1;
			frame2_kp->points.push_back(p);
		}

		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		viewer->addCoordinateSystem(1);
		viewer->initCameraParameters();
		viewer->setCameraPosition(0.4, 0.4, 0.5, 0, 0, 10, 0, 1, 0);

		viewer->addPointCloud(pointcloud_keypoints1, "keypoints");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypoints");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "keypoints");

		viewer->addPointCloud(frame2_kp, "keypoints2");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypoints2");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "keypoints2");


		viewer->addPointCloud(scene1, "scene1");
		viewer->addPointCloud(frame2, "scene2");

		for (int t = 0; t < graph_corps->size(); ++t)
		{
			int inx1 = graph_corps->at(t).index_query;
			int inx2 = graph_corps->at(t).index_match;
			std::stringstream ss_line;
			ss_line << "correspondence_line" << inx1 << "_" << inx2;
			pcl::PointXYZRGB& model_point = pointcloud_keypoints1->at(inx1);
			pcl::PointXYZRGB& scene_point = frame2_kp->at(inx2);

			viewer->addLine<pcl::PointXYZRGB, pcl::PointXYZRGB>(model_point, scene_point, 0, 255, 0, ss_line.str());
		}

		viewer->registerPointPickingCallback(point_picking_callback, (void*)&viewer);

		while (!viewer->wasStopped())
		{
			viewer->spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}*/


		//----------------------- show after apply pose----------------------------
		/*pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame1(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame2(new pcl::PointCloud<pcl::PointXYZRGB>);

		Eigen::Matrix4d mat1 = Eigen::Matrix4d::Identity();
		Eigen::Affine3f affine1(transformation_matrix.cast<float>());
		pcl::transformPointCloud(*scene1, *frame1, affine1);

		Eigen::Affine3f affine2(Eigen::Matrix4d::Identity().cast<float>());
		pcl::transformPointCloud(*scene2, *frame2, affine2);


		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2(new pcl::visualization::PCLVisualizer("3D Viewer2"));

		viewer2->setBackgroundColor(0, 0, 0);
		viewer2->addCoordinateSystem(1);
		viewer2->initCameraParameters();
		viewer2->setCameraPosition(0, 0, 5, 0, 0, -1, 0, 1, 0);

		viewer2->addPointCloud(frame1, "scene1");
		viewer2->addPointCloud(frame2, "scene2");

		while (!viewer2->wasStopped())
		{
		viewer2->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}*/
	}

	//save
	std::cout << "Save trajectory file...";
	traj.SaveToSPFile(traj_file);
	std::cout << "Done!\n";

	std::cout << "Save information file...";
	info.SaveToSPFile(info_file);
	std::cout << "Done!\n";

	return 0;
}
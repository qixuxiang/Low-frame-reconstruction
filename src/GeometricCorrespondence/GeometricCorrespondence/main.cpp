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
	std::string pointcloud_dir, pointcloud_ds_dir, keypoint_dir, descriptor_dir, camera_file, traj_file, info_file;

	if (argc != 9)
	{
		std::cout << "Usage:\n\GeometricCorrespondence.exe pointcloud_dir[input_dir] pointcloud_ds_dir[input_dir] keypoint_dir[input_dir] descriptor_dir[input_dir] camera_file[input_file] score_max_depth[<0: will ignore]"
			<< " traj_file[output_file] info_file[output_file]\n";
		return 0;
	}

	pointcloud_dir = argv[1];
	pointcloud_ds_dir = argv[2];
	keypoint_dir = argv[3];
	descriptor_dir = argv[4];
	camera_file = argv[5];
	score_max_depth = atof(argv[6]);

	traj_file = argv[7];
	info_file = argv[8];

	/*score_max_depth = -4.0;
	pointcloud_dir = "C:/Users/range/Desktop/ICL_NUIM_living_room/sandbox/pointcloud/";
	pointcloud_ds_dir = "C:/Users/range/Desktop/ICL_NUIM_living_room/sandbox/pointcloud_ds/";
	keypoint_dir = "C:/Users/range/Desktop/ICL_NUIM_living_room/sandbox/keypoint/";
	descriptor_dir = "C:/Users/range/Desktop/ICL_NUIM_living_room/sandbox/geometric_descriptor/";
	camera_file = "C:/Users/range/Desktop/ICL_NUIM_living_room/sandbox/camera_para.txt";

	traj_file = "C:/Users/range/Desktop/ICL_NUIM_living_room/sandbox/narf_traj.txt";
	info_file = "C:/Users/range/Desktop/ICL_NUIM_living_room/sandbox/narf_info.txt";*/


	int num_of_pc = std::count_if(
		boost::filesystem::directory_iterator(boost::filesystem::path(pointcloud_dir)),
		boost::filesystem::directory_iterator(),
		[](const boost::filesystem::directory_entry& e) {
		return e.path().extension() == ".pcd";  });

	// Load all downsample pointcloud
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> downsample_pc;
	for (int i = 0; i < num_of_pc; ++i)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_ds(new pcl::PointCloud<pcl::PointXYZRGB>);

		std::stringstream dsss1;
		dsss1 << pointcloud_ds_dir << "pointcloud_ds" << i << ".pcd";
		pcl::io::loadPCDFile(dsss1.str(), *scene_ds);
		downsample_pc.push_back(scene_ds);
	}

	// Load all keypoints
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> keypoints(num_of_pc);
	for (int i = 0; i < num_of_pc;++i)
	{
		std::stringstream pkss1;
		pkss1 << keypoint_dir << "keypoints" << i << ".pcd";
		if (!boost::filesystem::exists(pkss1.str()))
			continue;
		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoint(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::io::loadPCDFile(pkss1.str(), *keypoint);
		keypoints[i] = keypoint;
	}
	
	// Load all descriptors
	std::vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr> descriptors(num_of_pc);
	for (int i = 0; i < num_of_pc;++i)
	{
		ImageDescriptor dp;
		std::stringstream dpss;
		dpss << descriptor_dir << "descriptor" << i << ".txt";
		if (!boost::filesystem::exists(dpss.str()))
			continue;

		dp.LoadFromFile(dpss.str());
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptor(new pcl::PointCloud<pcl::FPFHSignature33>());
		for (int t = 0; t < dp.data_.size(); ++t)
		{
			pcl::FPFHSignature33 d;
			for (int k = 0; k < dp.data_[t].dvec.size(); ++k)
				d.histogram[k] = dp.data_[t].dvec[k];
			descriptor->points.push_back(d);
		}
		descriptors[i] = descriptor;
	}

	RGBDInformation info;	info.data_.resize(num_of_pc*num_of_pc);
	RGBDTrajectory  traj;	traj.data_.resize(num_of_pc*num_of_pc);
	
	/*int id1, id2;
	std::cout << "Input you want to see two image pose(Geometry)\n";
	std::cin >> id1 >> id2;*/

	// deal with corner correspondence
	if (boost::filesystem::exists(traj_file))
		boost::filesystem::remove(traj_file);
	if (boost::filesystem::exists(info_file))
		boost::filesystem::remove(info_file);
#pragma omp parallel for num_threads( 8 ) schedule( dynamic )
	for (int i = 0; i < num_of_pc; ++i)
	{
		for (int j = i + 1; j < num_of_pc; ++j)
		{
			std::stringstream ssf1, ssf2;
			ssf1 << keypoint_dir << "keypoints" << i << ".pcd";
			ssf2 << keypoint_dir << "keypoints" << j << ".pcd";

			if (!boost::filesystem::exists(ssf1.str()) || !boost::filesystem::exists(ssf2.str()))
				continue;

			int img1 = i;
			int img2 = j;

			/*if (img1 != id1 || img2 != id2)
				continue;*/

			std::cout << "Begin registration geometric correspondence between (" << img1 << ", " << img2 << ")\n";

			pcl::PointCloud<pcl::PointXYZ>::Ptr keypoint1 = keypoints[img1];
			pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptor1 = descriptors[img1];

			pcl::PointCloud<pcl::PointXYZ>::Ptr keypoint2 = keypoints[img2];
			pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptor2 = descriptors[img2];

			if (keypoint1->size() < 4 || keypoint2->size() < 4)
				continue;

			// guess correspondence
			std::cout << "Guess correspondences...";
			pcl::CorrespondencesPtr corps(new pcl::Correspondences());
			pcl::KdTreeFLANN<pcl::FPFHSignature33> match_search;
			match_search.setInputCloud(descriptor2);
			for (int t = 0; t < descriptor1->size(); ++t)
			{
				int N_nearest = 2;

				std::vector<int> neigh_indices(N_nearest);
				std::vector<float> neigh_sqr_dists(N_nearest);

				match_search.nearestKSearch(descriptor1->at(t), N_nearest, neigh_indices, neigh_sqr_dists);

				if (neigh_sqr_dists[0] < 0.8*neigh_sqr_dists[1])
				{
					pcl::Correspondence corr(t, neigh_indices[0], neigh_sqr_dists[0]);
					corps->push_back(corr);
				}
				/*for (int k = 0; k < N_nearest; ++k)
				{
				pcl::Correspondence corr(t, neigh_indices[k], neigh_sqr_dists[k]);
				corps->push_back(corr);
				}*/
			}
			std::cout << "Done!\n";

			if (corps->size() < 4)
				continue;

			//Graph matching
			std::cout << "Graph matching...";
			GraphMatching gm(*keypoint1, *keypoint2, *corps);
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
			//pcl::io::loadPCDFile(pcss1.str(), *scene1);			// multiple thread unsafe 

			//pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene2(new pcl::PointCloud<pcl::PointXYZRGB>);
			//std::stringstream pcss2;
			//pcss2 << pointcloud_dir << "pointcloud" << img2 << ".pcd";
			//pcl::io::loadPCDFile(pcss2.str(), *scene2);			// multiple thread unsafe 

			//---------------------show correspondence------------------------------
			//pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame2(new pcl::PointCloud<pcl::PointXYZRGB>);
			//for (int t = 0; t < scene2->points.size(); ++t)
			//{
			//	pcl::PointXYZRGB p = scene2->points[t];
			//	p.x += 3;
			//	p.z += 1;
			//	frame2->points.push_back(p);
			//}

			//pcl::PointCloud<pcl::PointXYZ>::Ptr frame2_kp(new pcl::PointCloud<pcl::PointXYZ>);
			//for (int t = 0; t < keypoint2->points.size(); ++t)
			//{
			//	pcl::PointXYZ p = keypoint2->points[t];
			//	p.x += 3;
			//	p.z += 1;
			//	frame2_kp->points.push_back(p);
			//}

			//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewer"));
			//viewer->setBackgroundColor(0, 0, 0);
			//viewer->addCoordinateSystem(1);
			//viewer->initCameraParameters();
			//viewer->setCameraPosition(0.4, 0.4, 0.5, 0, 0, 10, 0, 1, 0);

			//viewer->addPointCloud(keypoint1, "keypoints");
			//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypoints");
			//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "keypoints");

			//viewer->addPointCloud(frame2_kp, "keypoints2");
			//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "keypoints2");
			//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "keypoints2");


			//viewer->addPointCloud(scene1, "scene1");
			//viewer->addPointCloud(frame2, "scene2");

			////add correspondence line
			//for (int t = 0; t < graph_corps->size(); ++t)
			//{
			//	int inx1 = graph_corps->at(t).index_query;
			//	int inx2 = graph_corps->at(t).index_match;
			//	std::stringstream ss_line;
			//	ss_line << "correspondence_line" << inx1 << "_" << inx2;
			//	pcl::PointXYZ& model_point = keypoint1->at(inx1);
			//	pcl::PointXYZ& scene_point = frame2_kp->at(inx2);

			//	viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(model_point, scene_point, 0, 255, 0, ss_line.str());
			//}

			//viewer->registerPointPickingCallback(point_picking_callback, (void*)&viewer);

			//while (!viewer->wasStopped())
			//{
			//	viewer->spinOnce(100);
			//	boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			//}


			////----------------------- show after apply pose----------------------------
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
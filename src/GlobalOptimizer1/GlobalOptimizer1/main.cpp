#include <boost/filesystem.hpp>

#include "ProgressiveOpt.h"

int main(int argc, char** argv)
{
	double error_threhold;
	double score1, score2;
	double low_score_threshold;
	std::string pointcloud_dir, traj_file, info_file, pose_file, fail_file, selected_edge_file;

	if (argc != 11)
	{
		std::cout << "Usage: GlobalOptimizer1.exe\n" <<
			"pointcloud_dir[input dir]\n" <<
			"traj_file[input dir]\n" <<
			"info_file[input dir]\n" <<
			"pose_file[ouput_file]\n" <<
			"fail_file[ouput_file]\n" <<
			"selected_edge_file[output_file]\n"<<
			"low_score_threshold[input value, if geometric loop select edge below threshold, loop will be discarded]\n"<<
			"score1[input value, select consequence edge score threshold]\n" <<
			"score2[input value, select non-consequence edge score threshold]\n" <<
			"error_threhold[input value, if a dense environment, use a more rigorous value. Typically, around 100 frames use 0.005, around 200 frames use 0.0025 ]\n";
		return -1;
	}

	pointcloud_dir = argv[1];
	traj_file = argv[2];
	info_file = argv[3];
	pose_file = argv[4];
	fail_file = argv[5];
	selected_edge_file = argv[6];
	low_score_threshold = atof(argv[7]);
	score1 = atof(argv[8]);
	score2 = atof(argv[9]);
	error_threhold = atof(argv[10]);


	// init input file path
	//pointcloud_dir = "E:/Rec/ICL_NUIM_office/sandbox/pointcloud/";
	//traj_file = "E:/Rec/ICL_NUIM_office/sandbox/traj.txt";
	//info_file = "E:/Rec/ICL_NUIM_office/sandbox/info.txt";

	//low_score_threshold = 0.35;
	//score1 = 0.55;
	//score2 = 0.35;
	//error_threhold = 0.005;
	//// init output file path
	//pose_file = "C:/Users/range/Desktop/pose_opt1.txt";
	//fail_file = "C:/Users/range/Desktop/separate_model.txt";

	int frame_num = std::count_if(
		boost::filesystem::directory_iterator(boost::filesystem::path(pointcloud_dir)),
		boost::filesystem::directory_iterator(),
		[](const boost::filesystem::directory_entry& e) {
		return e.path().extension() == ".pcd";  });

	ProgressiveOpt opt(frame_num, traj_file, info_file, pose_file, fail_file, selected_edge_file, low_score_threshold, score1, score2, error_threhold);
	if (opt.Init())
		opt.Opt();
	
	//system("pause");
	return 0;
}
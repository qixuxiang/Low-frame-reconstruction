#include <Eigen/Eigen>
#include <vector>


struct FramedTransformation {
	int frame1_;
	int frame2_;
	Eigen::Matrix4d transformation_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	FramedTransformation(int frame1, int frame2, const Eigen::Matrix4d& t)
		: frame1_(frame1), frame2_(frame2), transformation_(t)
	{}
};

struct SortTraj{
	bool operator()(FramedTransformation& f1, FramedTransformation& f2)
	{
		if (f1.frame1_ < f2.frame1_)
			return true;
		if (f1.frame1_ > f2.frame1_)
			return false;
		if (f1.frame2_ < f2.frame2_)
			return true;
		else
			return false;
	}
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
		else
		{
			std::cout << "FILE CANNOT FOUND " << filename << "\n";
			exit(-1);
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


typedef Eigen::Matrix< double, 6, 6, Eigen::RowMajor > InformationMatrix;
struct FramedInformation {
	int frame1_;
	int frame2_;
	InformationMatrix information_;

	double score;
	int flag;			//0: harris3d  1:sift  2: corner

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	FramedInformation(int frame1, int frame2, const InformationMatrix& t)
		: frame1_(frame1), frame2_(frame2), information_(t), score(0.0)
	{}
	FramedInformation(int frame1, int frame2, const InformationMatrix& t, double sco)
		: frame1_(frame1), frame2_(frame2), information_(t), score(sco)
	{}
};

struct SortInfo{
	bool operator()(FramedInformation& f1, FramedInformation& f2)
	{
		if (f1.frame1_ < f2.frame1_)
			return true;
		else if (f1.frame1_ == f2.frame1_ && f1.frame2_<f2.frame2_)
			return true;
		else if (f1.frame1_ == f2.frame1_ && f1.frame2_==f2.frame2_ && f1.flag<f2.flag)
			return true;
		else
			return false;
	}
};


struct RGBDInformation {
	std::vector< FramedInformation, Eigen::aligned_allocator<FramedInformation> > data_;

	void LoadFromFile(std::string filename) {
		data_.clear();
		int frame1, frame2;
		double score;
		InformationMatrix info;
		FILE * f = fopen(filename.c_str(), "r");
		if (f != NULL) {
			char buffer[1024];
			while (fgets(buffer, 1024, f) != NULL) {
				if (strlen(buffer) > 0 && buffer[0] != '#') {
					sscanf(buffer, "%d %d %lf", &frame1, &frame2, &score);
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf %lf %lf %lf", &info(0, 0), &info(0, 1), &info(0, 2), &info(0, 3), &info(0, 4), &info(0, 5));
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf %lf %lf %lf", &info(1, 0), &info(1, 1), &info(1, 2), &info(1, 3), &info(1, 4), &info(1, 5));
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf %lf %lf %lf", &info(2, 0), &info(2, 1), &info(2, 2), &info(2, 3), &info(2, 4), &info(2, 5));
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf %lf %lf %lf", &info(3, 0), &info(3, 1), &info(3, 2), &info(3, 3), &info(3, 4), &info(3, 5));
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf %lf %lf %lf", &info(4, 0), &info(4, 1), &info(4, 2), &info(4, 3), &info(4, 4), &info(4, 5));
					fgets(buffer, 1024, f);
					sscanf(buffer, "%lf %lf %lf %lf %lf %lf", &info(5, 0), &info(5, 1), &info(5, 2), &info(5, 3), &info(5, 4), &info(5, 5));
					data_.push_back(FramedInformation(frame1, frame2, info, score));
				}
			}
			fclose(f);
		}
		else
		{
			std::cout << "FILE CANNOT FOUND " << filename << "\n";
			exit(-1);
		}
	}
	void SaveToFile(std::string filename) {
		FILE * f = fopen(filename.c_str(), "w");
		for (int i = 0; i < (int)data_.size(); i++) {
			InformationMatrix & info = data_[i].information_;
			fprintf(f, "%d\t%d\t%.8f\t%d\n", data_[i].frame1_, data_[i].frame2_, data_[i].score, data_[i].flag);
			fprintf(f, "%.8f %.8f %.8f %.8f %.8f %.8f\n", info(0, 0), info(0, 1), info(0, 2), info(0, 3), info(0, 4), info(0, 5));
			fprintf(f, "%.8f %.8f %.8f %.8f %.8f %.8f\n", info(1, 0), info(1, 1), info(1, 2), info(1, 3), info(1, 4), info(1, 5));
			fprintf(f, "%.8f %.8f %.8f %.8f %.8f %.8f\n", info(2, 0), info(2, 1), info(2, 2), info(2, 3), info(2, 4), info(2, 5));
			fprintf(f, "%.8f %.8f %.8f %.8f %.8f %.8f\n", info(3, 0), info(3, 1), info(3, 2), info(3, 3), info(3, 4), info(3, 5));
			fprintf(f, "%.8f %.8f %.8f %.8f %.8f %.8f\n", info(4, 0), info(4, 1), info(4, 2), info(4, 3), info(4, 4), info(4, 5));
			fprintf(f, "%.8f %.8f %.8f %.8f %.8f %.8f\n", info(5, 0), info(5, 1), info(5, 2), info(5, 3), info(5, 4), info(5, 5));
		}
		fclose(f);
	}
};
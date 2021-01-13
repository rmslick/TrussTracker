#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/embed.h> 
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <signal.h>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/centroid.h>

using MonoPoint = pcl::PointXYZ;
using MonoCloud = pcl::PointCloud<MonoPoint>;

namespace py = pybind11;
using namespace rs2;
using namespace cv;


class ObjectTracker
{
    private:
        pipeline pipe;
        pipeline_profile config;
        rs2_intrinsics color_intrinsics;
        rs2_intrinsics inrist;
        // Convert rs2::frame to cv::Mat
        cv::Mat FrameToMat( rs2::frame& f);
        // Define the function to be called when ctrl-c (SIGINT) is sent to process
        static void signal_callback_handler(int signum);
        void ClassifyAndWrite(Mat color_mat);
        void TrainingDataGen(Mat& color_mat, std::string& fPath);
        MonoCloud DFToCroppedPC(rs2::depth_frame  depth_frame, float * bl, float * tr);
        //Take object off the ground/table (Not used in this demo package)
        void PlanarExtraction(MonoCloud::Ptr& cloud, std::string fName);
        void EuclideanCluster(MonoCloud::Ptr& cloud, std::vector<MonoCloud>& clustersIn);
        void ScoreClusters(std::vector<MonoCloud>& clusters,std::vector<MonoCloud>& clusterIntersections,int& highestCluster);
        void InitCam();
        rs2::frameset GetFrames();
        void GetBBoxPoints(py::dict& rVal, depth_frame df ,MonoPoint& blXYZPoint, MonoPoint& trXYZPoint);
    public:
        ObjectTracker();
        ~ObjectTracker(){}
        void Run();

};
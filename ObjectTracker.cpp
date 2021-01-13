#include "ObjectTracker.h"


ObjectTracker::ObjectTracker()
{
    config = pipe.start();
    InitCam();
    signal(SIGINT, signal_callback_handler);
}
    // Convert rs2::frame to cv::Mat
cv::Mat ObjectTracker::FrameToMat( rs2::frame& f)
{
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();
    //std::cout << "FRAME TO MAT" << f.get_profile().format() << std::endl;
    if (f.get_profile().format() == RS2_FORMAT_BGR8)
    {
        return Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_RGB8)
    {
        auto r_rgb = Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
        Mat r_bgr;
        cvtColor(r_rgb, r_bgr, COLOR_RGB2BGR);
        return r_bgr;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16)
    {
        return Mat(Size(w, h), CV_16UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Y8)
    {
        return Mat(Size(w, h), CV_8UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32)
    {
        return Mat(Size(w, h), CV_32FC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");
}

void ObjectTracker::signal_callback_handler(int signum) {
    std::cout << "Caught signal " << signum <<" cleaning up gracefully." <<std::endl;
    // Terminate program
    destroyAllWindows();
    exit(signum);
}

void ObjectTracker::ClassifyAndWrite(Mat color_mat)
{
    imwrite("ColorFrame.jpg", color_mat);
    //imshow("Hello World",color_mat);
    //waitKey(1);
    //sleep(.1);
    py::module_ calc = py::module_::import("Actuator");

    py::object result = calc.attr("detect_truss")("ColorFrame.jpg");

    //sleep(.1);
}
void ObjectTracker::TrainingDataGen(Mat& color_mat, std::string& fPath)
{
    imwrite(fPath, color_mat);
    std::cout << "Saved to: " +fPath << std::endl;

}

MonoCloud ObjectTracker::DFToCroppedPC(rs2::depth_frame  depth_frame, float * bl, float * tr)
{
    rs2::points points;
    rs2::pointcloud pc;
    points = pc.calculate(depth_frame);
    auto vertices = points.get_vertices();
    std::vector<rs2::vertex> allPoints;
    //Crop the scan along z if near wall
    for (int i =0; i < points.size(); i++)
    {
        allPoints.push_back(vertices[i]);
    }
    MonoCloud::Ptr cloud( new MonoCloud);
    cloud->resize( allPoints.size() );
    // Filter for channel
    
    for(int i =0; i < allPoints.size(); i++)
    {
        if( (allPoints.at(i).x > bl[0] && allPoints.at(i).x < tr[0]) && (allPoints.at(i).y > bl[1] && allPoints.at(i).y < tr[1]) && (allPoints.at(i).z < 1.3) )
        {
            (*cloud)[i].x = allPoints.at(i).x;
            (*cloud)[i].y = allPoints.at(i).y;
            (*cloud)[i].z = allPoints.at(i).z;
        }
    }
    //pcl::io::savePCDFileASCII ("Slice.pcd", *cloud);
    std::cout << "Point cloud saved\n";
    return *cloud;
}

    //Take object off the ground/table
    void ObjectTracker::PlanarExtraction(MonoCloud::Ptr& cloud, std::string fName)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr convexHull(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr objects(new pcl::PointCloud<pcl::PointXYZ>);

        // Get the plane model, if present.
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::SACSegmentation<pcl::PointXYZ> segmentation;
        segmentation.setInputCloud(cloud);
        segmentation.setModelType(pcl::SACMODEL_PLANE);
        segmentation.setMethodType(pcl::SAC_RANSAC);
        segmentation.setDistanceThreshold(0.01);
        segmentation.setOptimizeCoefficients(true);
        pcl::PointIndices::Ptr planeIndices(new pcl::PointIndices);
        segmentation.segment(*planeIndices, *coefficients);

        if (planeIndices->indices.size() == 0)
            std::cout << "Could not find a plane in the scene." << std::endl;
        else
        {
            // Copy the points of the plane to a new cloud.
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(cloud);
            extract.setIndices(planeIndices);
            extract.filter(*plane);

            // Retrieve the convex hull.
            pcl::ConvexHull<pcl::PointXYZ> hull;
            hull.setInputCloud(plane);
            // Make sure that the resulting hull is bidimensional.
            hull.setDimension(2);
            hull.reconstruct(*convexHull);

            // Redundant check.
            if (hull.getDimension() == 2)
            {
                // Prism object.
                pcl::ExtractPolygonalPrismData<pcl::PointXYZ> prism;
                prism.setInputCloud(cloud);
                prism.setInputPlanarHull(convexHull);
                // First parameter: minimum Z value. Set to 0, segments objects lying on the plane (can be negative).
                // Second parameter: maximum Z value, set to 10cm. Tune it according to the height of the objects you expect.
                prism.setHeightLimits(0.0f, 0.1f);
                pcl::PointIndices::Ptr objectIndices(new pcl::PointIndices);

                prism.segment(*objectIndices);

                // Get and show all points retrieved by the hull.
                extract.setIndices(objectIndices);
                extract.filter(*objects);
                pcl::io::savePCDFileASCII("PlanarSegments/"+fName, *objects);
            }
            else 
            {
                std::cout << "The chosen hull is not planar." << std::endl;
            }
        } 
    }


    void ObjectTracker::EuclideanCluster(MonoCloud::Ptr& cloud, std::vector<MonoCloud>& clustersIn)
    {
        // kd-tree object for searches.
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        kdtree->setInputCloud(cloud);

        // Euclidean clustering object.
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> clustering;
        // Set cluster tolerance to 2cm (small values may cause objects to be divided
        // in several clusters, whereas big values may join objects in a same cluster).
        clustering.setClusterTolerance(0.02);
        // Set the minimum and maximum number of points that a cluster can have.
        clustering.setMinClusterSize(100);
        clustering.setMaxClusterSize(25000);
        clustering.setSearchMethod(kdtree);
        clustering.setInputCloud(cloud);
        std::vector<pcl::PointIndices> clusters;
        clustering.extract(clusters);

        // For every cluster...
        int currentClusterNum = 0;
        for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
        {
            // ...add all its points to a new cloud...
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
                cluster->points.push_back(cloud->points[*point]);
            cluster->width = cluster->points.size();
            cluster->height = 1;
            cluster->is_dense = true;

            // ...and save it to disk.
            if (cluster->points.size() <= 0)
                break;
            std::cout << "Cluster " << currentClusterNum << " has " << cluster->points.size() << " points." << std::endl;
            std::string fileName = "clusters/cluster" + std::to_string(currentClusterNum) + ".pcd";
            pcl::io::savePCDFileASCII(fileName, *cluster);
            clustersIn.push_back(*cluster);
            currentClusterNum++;
        }
    }
    void ObjectTracker::ScoreClusters(std::vector<MonoCloud>& clusters,std::vector<MonoCloud>& clusterIntersections,int& highestCluster)
    {
        Mat imgMask = imread("Masked.png");
        std::vector<Mat> three_channels;
        split(imgMask,three_channels);
        int highestScore = -1;
        highestCluster = -1;
        for (int i =0; i < clusters.size();i++)
        {
            MonoCloud cluster = clusters.at(i); 
            int score = 0;
            std::vector<MonoPoint> hitPoints;
            for (int j =0; j < cluster.size(); j++)
            {
                float point[3];
                point[0] = cluster[j].x;
                point[1] = cluster[j].y;
                point[2] = cluster[j].z;
                float pixel[2];
                //std::cout << "Projecting\n";
                rs2_project_point_to_pixel(pixel,&color_intrinsics,point);
                try
                {
                    int r = (int) three_channels[0].at<uchar>((int) pixel[1], (int)pixel[0]);
                    int g = (int) three_channels[1].at<uchar>((int) pixel[1], (int)pixel[0]);
                    int b = (int) three_channels[2].at<uchar>((int) pixel[1], (int)pixel[0]);
                    //Check for a hit
                    if(r == 255 && g == 255 && b == 255)
                    {
                        hitPoints.push_back(cluster[j]);
                        score += 1;
                    }
                    three_channels[0].at<uchar>((int) pixel[1], (int)pixel[0]) = 0;
                    three_channels[1].at<uchar>((int) pixel[1], (int)pixel[0]) = 255;
                    three_channels[2].at<uchar>((int) pixel[1], (int)pixel[0]) = 0;
                    //  std::cout << "Made it\n";

                }
                catch(...)
                {
                    std::cout << "Invalid point.\n";
                }
            }
            MonoCloud intersectionCloud;
            intersectionCloud.resize(hitPoints.size());
            for(int i = 0; i < hitPoints.size(); i++)
            {
                intersectionCloud.push_back(hitPoints.at(i));
            }
            clusterIntersections.push_back(intersectionCloud);
            std::cout << "Scored cluster at: " << score << std::endl;
            if (score > highestScore)
            {
                highestScore = score;
                highestCluster = i;
            }
            std::cout << "Cluster " << highestCluster << " is the winner.\n";
            Mat maskMerge;
            merge(three_channels,maskMerge);
            imwrite("MaskOverlaid.jpg",maskMerge);   
        }
    }
    //Wake up call to cam, filters out groggy pics
    void ObjectTracker::InitCam()
    {
        stream_profile profile = config.get_stream(RS2_STREAM_COLOR).as<video_stream_profile>();
        rs2::align align_to(RS2_STREAM_COLOR);
        for(int i =0; i < 15; i++)
        {
            auto data = pipe.wait_for_frames();
            data = align_to.process(data);
        }
    }
    rs2::frameset ObjectTracker::GetFrames()
    {
        stream_profile profile = config.get_stream(RS2_STREAM_COLOR).as<video_stream_profile>();
        rs2::colorizer color_map;
        rs2::align align_to(RS2_STREAM_COLOR);
        auto data = pipe.wait_for_frames();
        data = align_to.process(data);
        return data;
    }
    void ObjectTracker::GetBBoxPoints(py::dict& rVal, depth_frame df ,MonoPoint& blXYZPoint, MonoPoint& trXYZPoint)
    {
        
        float pixel1[2];
        float pixel2[2];
        
        pixel1[0] = rVal["x1"].cast<int>();
        pixel1[1] = rVal["y1"].cast<int>();

        pixel2[0] = rVal["x2"].cast<int>();
        pixel2[1] = rVal["y2"].cast<int>();

        float blpoint[3];
        float trpoint[3];

        float pixel_distance_in_meters1 = df.get_distance(rVal["x1"].cast<int>(), rVal["y1"].cast<int>());
        float pixel_distance_in_meters2 = df.get_distance(rVal["x2"].cast<int>(), rVal["y2"].cast<int>());
        
        rs2_deproject_pixel_to_point(blpoint, &inrist,pixel1,pixel_distance_in_meters1);
        rs2_deproject_pixel_to_point(trpoint,&inrist,pixel2,pixel_distance_in_meters2);

        blXYZPoint.x = blpoint[0];
        blXYZPoint.y = blpoint[1];
        blXYZPoint.z = blpoint[2];

        trXYZPoint.x = trpoint[0];
        trXYZPoint.y = trpoint[1];
        trXYZPoint.z = trpoint[2];
    }
    void ObjectTracker::Run()
    {

        py::scoped_interpreter guard{}; // start the interpreter and keep it alive
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("append")("/home/rmslick/.pyenv/versions/ObjectTracker/lib/python3.6/site-packages");
        py::module_ trussClass = py::module_::import("truss");
        std::cout << "Truss imported.\n";
        //Perform classification
        int counter = 0;
        while(true)
        {
            rs2::frameset data = GetFrames();
            auto color_frame = data.get_color_frame();
            auto depth_frame = data.get_depth_frame();
            inrist = rs2::video_stream_profile(depth_frame.get_profile()).get_intrinsics();
            color_intrinsics = rs2::video_stream_profile(color_frame.get_profile()).get_intrinsics();
            Mat color_mat = FrameToMat(color_frame);
            //NOTE: Passing a cv matrix to ClassifyCornerJoint is in progress. This change is
            //      will be added in the next push. See issues for more details.
            imwrite("scanCap.jpg",color_mat);
            //trussClass passes back a dictionary with bbox pairs
            py::dict rVal = (trussClass.attr("ClassifyCornerJoint")("mask_rcnn_truss_0030.h5","scanCap.jpg"));
            bool rv = rVal["found"].cast<bool>();
            if (rv)
            {
                Mat bBox = imread("BBox.png");
                MonoPoint blXYZPoint;
                MonoPoint trXYZPoint;

                GetBBoxPoints(rVal,depth_frame,blXYZPoint,trXYZPoint);
                try{
                    //Perform channel cropping
                    float blPoint[3];
                    float trPoint[3];

                    blPoint[0] = blXYZPoint.x;
                    blPoint[1] = blXYZPoint.y;
                    blPoint[2] = blXYZPoint.z;
                    
                    trPoint[0] = trXYZPoint.x;
                    trPoint[1] = trXYZPoint.y;
                    trPoint[2] = trXYZPoint.z;

                    MonoCloud croppedCloud = DFToCroppedPC(depth_frame,blPoint,trPoint);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
                    cloud = croppedCloud.makeShared();
                    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>);
                    // Preprocess the channel point cloud
                    pcl::VoxelGrid<pcl::PointXYZ> filter;
                    filter.setInputCloud(cloud);
                    filter.setLeafSize(0.01f, 0.01f, 0.01f);
                    filter.filter(*filteredCloud);
                    std::vector<MonoCloud> clusters; 
                    
                    //Cluster for objects in the channel
                    EuclideanCluster(filteredCloud, clusters);
                    //Score clusters
                    std::vector<MonoCloud> clusterIntersections;
                    int highestCluster;
                    ScoreClusters(clusters,clusterIntersections, highestCluster);
                    pcl::io::savePCDFileASCII("IntersectedCluster.pcd", clusterIntersections.at(highestCluster));
                    Eigen::Vector4f centroid;
                    pcl::compute3DCentroid(clusterIntersections.at(highestCluster), centroid);

                    std::cout << "Bottom left " << blPoint[0]<<" " << blPoint[1] <<" "<< blPoint[2]<<std::endl;
                    std::cout << "Top right " << trPoint[0]<<" " << trPoint[1] <<" "<< trPoint[2] <<std::endl;
                    std::string locationString = "Centroid at: " + std::to_string(centroid[0]) + " " + std::to_string(centroid[1]) + " " +std::to_string(centroid[2]);
                    cv::putText(bBox, locationString, cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX,1.0,CV_RGB(0, 255, 0), 2);

                    imshow("image",bBox);
                    waitKey(0);
                }
                catch(...)
                {
                    std::cout << "Error cropping.\n";
                }
            }
            
            //ClassifyAndWrite(color_mat);
            //std::string path = "TrainingData/"+std::to_string(counter)+".jpg";
            //TrainingDataGen(color_mat,path);
            counter+=1;
        }
    }
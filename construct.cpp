#include <iostream>
#include <algorithm>
#include <string>
using namespace std;
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <pcl/common/transforms.h>
//#include <pcl/visualization/cloud_viewer.h>

// Eigen !
#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace cv;
#include <iostream>
#include <string> 
#include <fstream>
#include <sstream>
#include <fstream>
#include <vector>
using namespace std;

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>


//g2o的头文件
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
// 定义点云类型
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud; 

typedef g2o::BlockSolver_6_3 SlamBlockSolver; 
typedef g2o::LinearSolverEigen< SlamBlockSolver::PoseMatrixType > SlamLinearSolver; 
bool check_loop = false;
struct RESULT_OF_PNP{
    cv::Mat rvec, tvec;
    int inliers;
};
int pic_cnt = 0;
int pic_start = 0;
std::map<pair<int, int>, double> judg;

std::vector<vector<KeyPoint> > key_points_l;//used to do reconstruction
std::vector<vector<KeyPoint> > key_points_l_l;

std::vector<vector<KeyPoint> > key_points_r;

std::vector<Mat> descriptors_l;
std::vector<Mat> descriptors_l_l;

std::vector<Mat> descriptors_r;


g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct( "Cauchy" );

struct FRAME{
    cv::Mat rgb;
    cv::Mat disparity;
    cv::Mat mask;
    int frameID;
    std::map<pair<int, int>, Point3f> depth; 

};

struct CAMERA_MODEL{
	double fx, fy;
	double cx, cy;
	double baseline;
    double disp;
    int cols;
    int rows;
};

CAMERA_MODEL left_c;

std::vector<FRAME*> keyframes;

void cal_left(int id, Mat img){
        Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);

        vector<KeyPoint>  key_points1_r, key_points1_l;

        Mat descriptors1_l, descriptors1_r, mascara;

        Mat img_l = img(Rect(0,0,img.cols/2, img.rows));


        sift->detectAndCompute(img, mascara, key_points1_r, descriptors1_r);
        key_points_l.push_back(key_points1_r);
        descriptors_l.push_back(descriptors1_r);

        sift->detectAndCompute(img_l, mascara, key_points1_l, descriptors1_l);
        key_points_l_l.push_back(key_points1_l);
        descriptors_l_l.push_back(descriptors1_l);
}

void cal_right(Mat img){
        Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);

        vector<KeyPoint> key_points1;

        Mat descriptors1, mascara;

        sift->detectAndCompute(img, mascara, key_points1, descriptors1);
        key_points_r.push_back(key_points1);
        descriptors_r.push_back(descriptors1);
}

double normofTransform( cv::Mat rvec, cv::Mat tvec ){
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}

void get_file_names(std::vector<string> & names, std::vector<string> &names2){
    for(int i = pic_start; i < pic_cnt; i++){
        char name_buf[50];
        memset(name_buf,0,50);
        sprintf(name_buf,"/home/sgeoghy/slam/right_p/frame%06d.jpg",i);
        Mat right = imread(name_buf);
        cal_right(right);

        names2.push_back((string)name_buf);
        memset(name_buf,0,50);
        sprintf(name_buf,"/home/sgeoghy/slam/left_p/frame%06d.jpg",i);
        Mat left = imread(name_buf);
        cal_left(i,left);
        names.push_back((string)name_buf);
    } 
}


Eigen::Isometry3d cvMat2Eigen(cv::Mat rvec, cv::Mat tvec )
{
    cv::Mat R;
    cv::Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    cv::cv2eigen(R,r);
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd angle(r);
    T = angle;
    T(0,3) = tvec.at<double>(0,0); 
    T(1,3) = tvec.at<double>(1,0); 
    T(2,3) = tvec.at<double>(2,0);
    //cout << rvec << endl << R << endl << tvec << endl;
    //cout << T.matrix() << endl;

    //cout << T.inverse().matrix() << endl;
    //cin.get();
     return T;
}

void extract_key(std::vector<KeyPoint> qu, std::vector<KeyPoint> &tr, Mat descriptors_q, Mat &descriptors_r){
    for(int i = 0; i < qu.size(); i++){
        if(qu[i].pt.x > left_c.cols/2)
            tr.push_back(qu[i]);        
    }

    descriptors_r = Mat::zeros(tr.size(), descriptors_q.cols, CV_64FC1);
    int index = 0;
    for(int i = 0; i < qu.size(); i++){
        if(qu[i].pt.x > left_c.cols/2){
            Rect r(0,i,128,1);
            Mat dstroi = descriptors_r(Rect(0,index,128,1));
            descriptors_q(r).convertTo(dstroi, dstroi.type(), 1, 0);
        }
    }
}


void calMatch(int left, int right, bool sfm, std::vector<KeyPoint> &qu, std::vector<KeyPoint>&tr){
        //sfm means to calculate matching points in subsequent frames using left half of the picture
        //!sfm means to calculate depth information of the whole picture

        vector<KeyPoint> key_points1, key_points2;

        Mat descriptors1, descriptors2, mascara;

        if(sfm){
            key_points1 = key_points_l_l[left];

            key_points2 = key_points_l_l[right];            
            descriptors1 = descriptors_l_l[left];

            descriptors2 = descriptors_l_l[right];

        }
        else{
            key_points1 = key_points_l[left];

            key_points2 = key_points_r[right];            
            descriptors1 = descriptors_l[left];

            descriptors2 = descriptors_r[right];
        }


        vector<cv::DMatch>matches;
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
        matcher->match(descriptors1, descriptors2, matches);


        vector<KeyPoint> R_keypoint01, R_keypoint02;
        for (size_t i = 0; i < matches.size(); i++)
        {
            R_keypoint01.push_back(key_points1[matches[i].queryIdx]);
            R_keypoint02.push_back(key_points2[matches[i].trainIdx]);
        }

        vector<Point2f>p01, p02;
        for (size_t i = 0; i < matches.size(); i++)
        {
            p01.push_back(R_keypoint01[i].pt);
            p02.push_back(R_keypoint02[i].pt);
        }

        vector<uchar> RansacStatus;
        Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus,  FM_RANSAC);

        vector<KeyPoint> RR_keypoint01, RR_keypoint02;

        for (size_t i = 0; i < matches.size(); i++)
        {
            if (RansacStatus[i] != 0)
            {
                RR_keypoint01.push_back(R_keypoint01[i]);
                RR_keypoint02.push_back(R_keypoint02[i]);
                /*
                if(sfm){
                    int ind1 = l2g[left][matches[i].trainIdx];
                    int ind2 = l2g[right][matches[i].queryIdx];
                    _find(ind1);
                    _find(ind2);
                    if(par[ind1] == par[ind2])continue;
                    if(par[ind2] == ind2)
                        par[ind2] = ind1;
                    else if(par[ind1] == ind1)
                        par[ind1] = ind1;
                    else
                        unite(ind1, ind2);
                }
                */
            }
        }

        for (int i = 0; i < RR_keypoint01.size(); i++){

            qu.push_back(RR_keypoint01.at(i));
            tr.push_back(RR_keypoint02.at(i));

        }


}

double calP(RESULT_OF_PNP result, std::map<pair<int, int>, Point3f>&prev_pt,std::map<pair<int, int>, Point3f>&after_pt, Mat prev, Mat now, Mat mask, Mat mask2, vector< cv::KeyPoint > &kp1, vector< cv::KeyPoint > &kp2){
    cv::Mat Q = Mat::zeros(4, 4, CV_64FC1);
    cv::Mat P = Mat::zeros(4, 1, CV_64FC1);
    cv::Mat R;
    cv::Rodrigues( result.rvec, R );
    Eigen::Matrix3d r;
    cv::cv2eigen(R,r);
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd angle(r);
    T = angle;
    T(0,3) = result.tvec.at<double>(0,0); 
    T(1,3) = result.tvec.at<double>(1,0); 
    T(2,3) = result.tvec.at<double>(2,0);
    T = T.inverse();
    Q.at<double>(0,0) = T(0,0);
    Q.at<double>(0,1) = T(0,1);
    Q.at<double>(0,2) = T(0,2);
    Q.at<double>(0,3) = T(0,3);
    Q.at<double>(1,0) = T(1,0);
    Q.at<double>(1,1) = T(1,1);
    Q.at<double>(1,2) = T(1,2);
    Q.at<double>(1,3) = T(1,3);
    Q.at<double>(2,0) = T(2,0);
    Q.at<double>(2,1) = T(2,1);
    Q.at<double>(2,2) = T(2,2);
    Q.at<double>(2,3) = T(2,3);
    Q.at<double>(3,0) = T(3,0);
    Q.at<double>(3,1) = T(3,1);
    Q.at<double>(3,2) = T(3,2);
    Q.at<double>(3,3) = T(3,3);

    //cout << Q << endl;

    double scale = 0.0;
    int cnt = 0;
    for(int i = 0; i < kp1.size(); i++){

        if(abs(mask.at<uchar>((int)kp1.at(i).pt.y, (int)kp1.at(i).pt.x)) < 1)
            continue;
        if(abs(mask2.at<uchar>((int)kp2.at(i).pt.y, (int)kp2.at(i).pt.x)) < 1)
            continue;

        //cout << prev_pt[make_pair((int)kp1.at(i).pt.y, (int)kp1.at(i).pt.x)] << endl;
        cnt += 1;


        Point3f q;

        q.x = after_pt[make_pair((int)kp2.at(i).pt.y, (int)kp2.at(i).pt.x)].x;
        q.y = after_pt[make_pair((int)kp2.at(i).pt.y, (int)kp2.at(i).pt.x)].y;
        q.z = after_pt[make_pair((int)kp2.at(i).pt.y, (int)kp2.at(i).pt.x)].z;
        P.at<double>(0,0) = q.x;
        P.at<double>(1,0) = q.y;
        P.at<double>(2,0) = q.z;
        P.at<double>(3,0) = 1.0;

        q.x = prev_pt[make_pair((int)kp1.at(i).pt.y, (int)kp1.at(i).pt.x)].x;
        q.y = prev_pt[make_pair((int)kp1.at(i).pt.y, (int)kp1.at(i).pt.x)].y;
        q.z = prev_pt[make_pair((int)kp1.at(i).pt.y, (int)kp1.at(i).pt.x)].z;


        //cout << P.at<double>(0,0)/q.x << ' ' << P.at<double>(1,0)/q.y << ' ' << P.at<double>(2,0)/q.z << endl;
        P = Q * P;

        //cout << P.at<double>(0,0)/q.x << ' ' << P.at<double>(1,0)/q.y << ' ' << P.at<double>(2,0)/q.z << endl;
        scale += P.at<double>(0,0)/q.x + P.at<double>(1,0)/q.y + P.at<double>(2,0)/q.z;
        

        //cout << P.at<double>(0,0) / q.x << ' ' << P.at<double>(1,0) / q.y << ' ' << P.at<double>(2,0) / q.z << endl;
    }
    //cin.get();
    if(cnt > 0)
        return scale / (3.0 * cnt);
    else
        return 1.0;

}

RESULT_OF_PNP calPNP(int id1, int id2, std::map<pair<int, int>, Point3f>&prev_pt, Mat prev, Mat now, Mat mask, vector< cv::KeyPoint > &kp1, vector< cv::KeyPoint > &kp2){

	
	calMatch(id1, id2, true, kp1, kp2);

    // 第一个帧的三维点
    vector<cv::Point3f> pts_obj;
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;

    Mat show_p = prev.clone();
    Mat show_q = now.clone();


	for(int i = 0; i < kp1.size(); i++){

		if(abs(mask.at<uchar>((int)kp1.at(i).pt.y, (int)kp1.at(i).pt.x)) < 1)
			continue;

        pts_img.push_back(kp2.at(i).pt);
		pts_obj.push_back(prev_pt[make_pair((int)kp1.at(i).pt.y, (int)kp1.at(i).pt.x)]);
                //cout << prev_pt[make_pair((int)kp1.at(i).pt.y, (int)kp1.at(i).pt.x)] << endl;
	}
    

    cv::Mat Q = Mat::zeros(3, 3, CV_64FC1); 
    Q.at<double>(0,0) = left_c.fx;
    Q.at<double>(0, 2) = left_c.cx;
    Q.at<double>(1,1) = left_c.fy;
    Q.at<double>(1,2) = left_c.cy;
    Q.at<double>(2,2) = 1;

    cv::Mat rvec, tvec, inliers;
    cv::Mat distCoeffs(4,1,CV_64FC1);
    distCoeffs.at<double>(0) = 0.0;
    distCoeffs.at<double>(1) = 0.0;
    distCoeffs.at<double>(2) = 0.0;
    distCoeffs.at<double>(3) = 0.0;  
    if(pts_obj.size() < 15){



        rvec = Mat::zeros(3, 1, CV_64FC1);
        rvec.at<double>(0,0) = 100000;
        rvec.at<double>(1,0) = 100000;
        rvec.at<double>(2,0) = 100000;
        
        RESULT_OF_PNP result;
        result.rvec = rvec;
        result.tvec = tvec;
        result.inliers = inliers.rows;
        result.tvec = Mat::zeros(3, 1, CV_64FC1);
        
        return result;
    }
    cv::solvePnPRansac( pts_obj, pts_img, Q, cv::Mat(), rvec, tvec, false, 100, 2, 0.99, inliers );

    cout<<"inliers: "<<inliers.rows<<endl;
    //cout<<"R="<<rvec<<endl;
    //cout<<"t="<<tvec<<endl;    


    RESULT_OF_PNP result;
    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;

    return result;
}

void get_color(g2o::SparseOptimizer &globalOptimizer){
    cv::Mat Q = Mat::zeros(3, 3, CV_64FC1); 
    Q.at<double>(0,0) = left_c.fx;
    Q.at<double>(0, 2) = left_c.cx;
    Q.at<double>(1,1) = left_c.fy;
    Q.at<double>(1,2) = left_c.cy;
    Q.at<double>(2,2) = 1;


    cv::Mat distCoeffs(4,1,CV_64FC1);
    distCoeffs.at<double>(0) = 0.0;
    distCoeffs.at<double>(1) = 0.0;
    distCoeffs.at<double>(2) = 0.0;
    distCoeffs.at<double>(3) = 0.0;  
    
    Eigen::Isometry3d pose;
    std::vector<cv::Point3f> objectPoints;
    std::map<pair<int, int>, Point3f>::iterator it;
    std::vector<cv::Point2f> projectedPoints;
    


    for(int index = 0; index < keyframes.size(); index++){
        Mat color = keyframes[index]->rgb;
        std::map<pair<int, int>, Point3f>prev_pt = keyframes[index]->depth;

        Mat rvec = Mat::zeros(3, 3, CV_64FC1);
        Mat tvec = Mat::zeros(3, 1, CV_64FC1);

        for(int ud = index-1; ud >= index - 7; ud--){
            if(ud < 0)break;

            pose = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[ud]->frameID ))->estimate().inverse() * 
                dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[index]->frameID ))->estimate();

            for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                rvec.at<double>(i,j) = pose(i,j);


            Mat frame = keyframes[ud]->rgb;
            tvec.at<double>(0,0) = pose(0,3);
            tvec.at<double>(1,0) = pose(1,3);
            tvec.at<double>(2,0) = pose(2,3);

            for(it=prev_pt.begin();it!=prev_pt.end();++it){
                objectPoints.clear();
                projectedPoints.clear();
    
                objectPoints.push_back(it->second);

                cv::projectPoints(objectPoints, rvec, tvec, Q, distCoeffs, projectedPoints);

                if(int(projectedPoints[0].y) < 0 || int(projectedPoints[0].y) >= left_c.rows)continue;
                if(int(projectedPoints[0].x) < 0 || int(projectedPoints[0].x) >= left_c.cols/2)continue;
                
                if(ud == index - 1)
                    color.at<Vec3b>(it->first.first, it->first.second) =  frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x));
        
                else
                    color.at<Vec3b>(it->first.first, it->first.second) =  color.at<Vec3b>(it->first.first, it->first.second) * 0.3 + 0.7 * frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x));
            
                circle(color, Point2f(it->first.second, it->first.first), 2, Scalar(frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x))[0],frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x))[1],frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x))[2]),-1);
            }


        }

        for(int ud = index+1; ud <= index + 7; ud++){

            if(ud >= keyframes.size())break;
            pose = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[ud]->frameID ))->estimate().inverse() * 
                dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[index]->frameID ))->estimate();
            for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                rvec.at<double>(i,j) = pose(i,j);


            Mat frame = keyframes[ud]->rgb;
            tvec.at<double>(0,0) = pose(0,3);
            tvec.at<double>(1,0) = pose(1,3);
            tvec.at<double>(2,0) = pose(2,3);
            for(it=prev_pt.begin();it!=prev_pt.end();++it){
                objectPoints.clear();
                projectedPoints.clear();
    
                objectPoints.push_back(it->second);

                cv::projectPoints(objectPoints, rvec, tvec, Q, distCoeffs, projectedPoints);

                if(int(projectedPoints[0].y) < 0 || int(projectedPoints[0].y) >= left_c.rows)continue;
                if(int(projectedPoints[0].x) < 0 || int(projectedPoints[0].x) >= left_c.cols/2)continue;
                
                if(ud == 1)
                    color.at<Vec3b>(it->first.first, it->first.second) =  frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x));
        
                else
                    color.at<Vec3b>(it->first.first, it->first.second) =  color.at<Vec3b>(it->first.first, it->first.second) * 0.3 + 0.7 * frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x));
            
                circle(color, Point2f(it->first.second, it->first.first), 2, Scalar(frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x))[0],frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x))[1],frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x))[2]),-1);
            }


        }        

        srand( (unsigned int) time(NULL) );
    
        if ( keyframes.size() > 10 )
        {

            for(int kd = 0; kd < 10; kd++){
                int ud = rand()%keyframes.size();
                if(ud == index)continue;

                pose = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[ud]->frameID ))->estimate().inverse() * 
                    dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[index]->frameID ))->estimate();
                for(int i = 0; i < 3; i++)
                    for(int j = 0; j < 3; j++)
                        rvec.at<double>(i,j) = pose(i,j);


                Mat frame = keyframes[ud]->rgb;
                tvec.at<double>(0,0) = pose(0,3);
                tvec.at<double>(1,0) = pose(1,3);
                tvec.at<double>(2,0) = pose(2,3);
                for(it=prev_pt.begin();it!=prev_pt.end();++it){
                    objectPoints.clear();
                    projectedPoints.clear();
    
                    objectPoints.push_back(it->second);

                    cv::projectPoints(objectPoints, rvec, tvec, Q, distCoeffs, projectedPoints);

                    if(int(projectedPoints[0].y) < 0 || int(projectedPoints[0].y) >= left_c.rows)continue;
                    if(int(projectedPoints[0].x) < 0 || int(projectedPoints[0].x) >= left_c.cols/2)continue;
                
                    color.at<Vec3b>(it->first.first, it->first.second) =  color.at<Vec3b>(it->first.first, it->first.second) * 0.3 + 0.7 * frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x));
            
                    circle(color, Point2f(it->first.second, it->first.first), 2, Scalar(frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x))[0],frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x))[1],frame.at<Vec3b>(int(projectedPoints[0].y), int(projectedPoints[0].x))[2]),-1);
                }


            }


    
        }

        keyframes[index]->rgb = color;
    }


}

void checkKeyframes( g2o::SparseOptimizer &globalOptimizer, int prev_id, int curr_id){

    FRAME *prev = keyframes[prev_id];
    FRAME* curr = keyframes[curr_id];

    std::vector<KeyPoint> kp1, kp2;

    RESULT_OF_PNP result = calPNP( prev->frameID, curr->frameID, prev->depth, prev->rgb, curr->rgb, prev->mask.clone(), kp1,kp2);

    double norm = normofTransform(result.rvec, result.tvec);

    double norm2 = calP(result, prev->depth, curr->depth, prev->rgb, curr->rgb, prev->mask, curr->mask, kp1,kp2);

    if(norm > 1 || norm2 < 0.95 || norm2 > 1.05)return;
    if(norm < 0.001)return;

    //cout << "norm is: " << norm << endl;

    //cout << "add an edge: " << prev->frameID << ' ' << curr->frameID << endl;

    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    // 连接此边的两个顶点id

    if(globalOptimizer.vertex(prev->frameID) == NULL || globalOptimizer.vertex(curr->frameID) == NULL){
        cout << "NULL POINTER\n";
        return;
    }

    edge->setVertex( 0, globalOptimizer.vertex(prev->frameID ));
    edge->setVertex( 1, globalOptimizer.vertex(curr->frameID ));

    edge->setRobustKernel( robustKernel  );
    
    // 信息矩阵
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
    // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
    // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
    // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
    information(0,0) = information(1,1) = information(2,2) = 1000.0;
    information(3,3) = information(4,4) = information(5,5) = 1000.0;
    // 也可以将角度设大一些，表示对角度的估计更加准确
    edge->setInformation( information );
    // 边的估计即是pnp求解之结果
    Eigen::Isometry3d T = cvMat2Eigen(result.rvec.clone(), result.tvec.clone() );
    // edge->setMeasurement( T );
    edge->setMeasurement( T.inverse() );
    // 将此边加入图中
    globalOptimizer.addEdge(edge);

    cout << "add an edge: " << prev->frameID << ' ' << curr->frameID << endl;


}
void checkRandomLoops( g2o::SparseOptimizer &globalOptimizer, FRAME* currFrame)
{

    static int random_loops = 5;
    srand( (unsigned int) time(NULL) );
    
    if ( keyframes.size() <= random_loops )
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<keyframes.size(); i++)
        {
            if(keyframes.at(i)->frameID == currFrame->frameID)continue;
            checkKeyframes(globalOptimizer, i, keyframes.size() - 1);
        }
    }
    else
    {
        // randomly check loops
        for (int i=0; i<random_loops; i++)
        {
            int index = rand()%keyframes.size();
            if(keyframes.at(index)->frameID == currFrame->frameID)continue;
            checkKeyframes(globalOptimizer, index, keyframes.size() - 1);
        }
    }
}


void checkNearbyLoops( g2o::SparseOptimizer &globalOptimizer, FRAME* currFrame)
{

    static int nearby_loops = 5;
    
    if ( keyframes.size() <= nearby_loops )
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<keyframes.size(); i++)
        {
           
            if(keyframes.at(i)->frameID == currFrame->frameID)continue;
            checkKeyframes(globalOptimizer, i, keyframes.size() - 1);
        }
    }
    else
    {
        // check the nearest ones
        for (size_t i = keyframes.size()-nearby_loops; i<keyframes.size(); i++)
        {
            if(keyframes.at(i)->frameID == currFrame->frameID)continue;
            checkKeyframes(globalOptimizer, i, keyframes.size() - 1);
        }
    }
}


bool checkframes(g2o::SparseOptimizer &globalOptimizer, FRAME* curr, FRAME* prev){
    std::vector<KeyPoint> kp1, kp2;

    RESULT_OF_PNP result = calPNP( prev->frameID, curr->frameID, prev->depth, prev->rgb, curr->rgb, prev->mask.clone(), kp1,kp2);

    double norm = normofTransform(result.rvec, result.tvec);
    double norm2 = calP(result, prev->depth, curr->depth, prev->rgb, curr->rgb, prev->mask, curr->mask, kp1,kp2);



    //cout << "norm is: " << norm << endl;
    if(norm > 1 || norm2 > 1.05 || norm2 < 0.95)return false;
    //cout << "add an edge: " << prev->frameID << ' ' << curr->frameID << endl;

    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    // 连接此边的两个顶点id

    if(globalOptimizer.vertex(prev->frameID) == NULL){
        cout << "NULL POINTER\n";
        return false;
    }

        Eigen::Isometry3d T = cvMat2Eigen(result.rvec.clone(), result.tvec.clone() );
    if(globalOptimizer.vertex(curr->frameID) == NULL){
        g2o::VertexSE3 *v = new g2o::VertexSE3();
    
        v->setId(curr->frameID);
        //g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( prev->frameID ));    

        v->setEstimate( Eigen::Isometry3d::Identity() );
        //v->setEstimate(vertex->estimate() * T.inverse());

        globalOptimizer.addVertex(v);
    }
    edge->setVertex( 0, globalOptimizer.vertex(prev->frameID ));
    edge->setVertex( 1, globalOptimizer.vertex(curr->frameID ));

    edge->setRobustKernel( robustKernel  );
    // 信息矩阵
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
    // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
    // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
    // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
    information(0,0) = information(1,1) = information(2,2) = 1000.0;
    information(3,3) = information(4,4) = information(5,5) = 1000.0;
    // 也可以将角度设大一些，表示对角度的估计更加准确
    edge->setInformation( information );
    // 边的估计即是pnp求解之结果

    // edge->setMeasurement( T );
    edge->setMeasurement( T.inverse() );
    // 将此边加入图中
    globalOptimizer.addEdge(edge);

    cout << "add an edge: " << prev->frameID << ' ' << curr->frameID << endl;
    return true;
}

void check_exist( g2o::SparseOptimizer &globalOptimizer, FRAME* currFrame){
      static int random_loop = 7;
      bool flag = false;
    srand( (unsigned int) time(NULL) );
    
    if ( keyframes.size() <= random_loop )
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<keyframes.size(); i++)
        {
            if(keyframes.at(i)->frameID == currFrame->frameID)continue;
            if(checkframes(globalOptimizer, currFrame, keyframes.at(i)))flag = true;
        }


    }
    else
    {
        // randomly check loops
        for (int i=0; i<random_loop; i++)
        {
            int index = rand()%keyframes.size();
            if(keyframes.at(index)->frameID == currFrame->frameID)continue;
            if(checkframes(globalOptimizer, currFrame, keyframes.at(index)))flag = true;
        }
        
        for (int i=0; i<random_loop; i++)
        {
            int index = keyframes.size() - i - 1;
            if(keyframes.at(index)->frameID == currFrame->frameID)continue;
            if(checkframes(globalOptimizer, currFrame, keyframes.at(index)))flag = true;
        }

    }
    if(flag){
        keyframes.push_back(currFrame); 
        cout << "--------------exist before------------" << endl;
    }
    else
        cout << "--------------definitely bad frame----------------" << endl;

}


void get_depth( g2o::SparseOptimizer& globalOptimizer, int index, string left, string right){

    cv::Mat img_left = cv::imread(left);

    cv::Mat img_right = cv::imread(right);


    vector<KeyPoint>qu, tr;
    calMatch(index, index, false, qu, tr);
    Mat left_ds = img_left.clone();
    Mat right_ds = img_right.clone();
/*
    for(int i = 0; i < qu.size(); i++){
        circle(left_ds, Point2f(qu.at(i).pt.x, qu.at(i).pt.y), 5, Scalar(0,0,255),-1);
        circle(right_ds, Point2f(tr.at(i).pt.x, tr.at(i).pt.y), 5, Scalar(0,0,255),-1);

    }
        imshow("left", left_ds);
        imshow("right", right_ds);
        waitKey(0);
*/

    //cout << qu.size() << ' ' << tr.size() << endl;
    Mat after = Mat::zeros(left_c.rows, left_c.cols, CV_8UC1);
    cv::Mat Q = Mat::zeros(4, 4, CV_64FC1); 
    Q.at<double>(0,0) = 1;
    Q.at<double>(0, 3) = -left_c.cx;
    Q.at<double>(1,1) = 1;
    Q.at<double>(1,3) = -left_c.cy;
    Q.at<double>(2,3) = (left_c.fx+left_c.fy)/2;
    Q.at<double>(3,2) = -1/left_c.baseline;  
    Q.at<double>(3,3) = left_c.disp / left_c.baseline;
    cv::Mat pix = Mat::zeros(4, 1, CV_64FC1);  
    cv::Mat pixW = Mat::zeros(4, 1, CV_64FC1);

    PointCloud::Ptr cloud2 ( new PointCloud );
    cv::Mat disp = Mat::zeros(left_c.rows, left_c.cols, CV_64FC1);

    std::map<pair<int, int>, Point3f>after_pt;

    //cin.get();
	for (int i = 0; i < qu.size(); i++){
        double d = qu.at(i).pt.x - tr.at(i).pt.x;

        disp.at<double>(int(qu.at(i).pt.y), int(qu.at(i).pt.x)) = d;

        pix.at<double>(0,0) = int(qu.at(i).pt.x);
        pix.at<double>(1,0) = int(qu.at(i).pt.y);
        pix.at<double>(2,0) = d;
        pix.at<double>(3,0) = 1;
        
        pixW = Q * pix;

        PointT p;
        p.z = -pixW.at<double>(2,0)/pixW.at<double>(3,0);
        p.x = -pixW.at<double>(0,0)/pixW.at<double>(3,0);
        p.y = -pixW.at<double>(1,0)/pixW.at<double>(3,0);

        cloud2->points.push_back(p);
        after.at<uchar>((int)qu.at(i).pt.y, (int)qu.at(i).pt.x) = 10;

        after_pt[make_pair((int)qu.at(i).pt.y, (int)qu.at(i).pt.x)] = Point3f(p.x, p.y, p.z);
    }

    if(keyframes.size() == 0){
        FRAME *key = new FRAME();
        key->mask = after;
        key->frameID = index;
        key->rgb = img_left.clone();
        Vec3b p;
        p[0] = 0;
        p[1] = 0;
        p[2] = 0;
        for(int i = 0; i < img_left.rows; i++)
            for(int j = left_c.cols/2; j < left_c.cols; j++)
                key->rgb.at<Vec3b>(i,j) = p;

        key->disparity = disp.clone();
        key->depth.insert(after_pt.begin(), after_pt.end());
        after_pt.clear();
        keyframes.push_back(key);

        return;
    }


    FRAME *key = keyframes[keyframes.size() - 1];
    Mat before, aft;

    before = key->rgb.clone();

   	aft = img_left.clone(); //(Rect(0,0,left_c.cols/2,left_c.rows)); 

    std::vector<KeyPoint> kp1, kp2;

    RESULT_OF_PNP result = calPNP(key->frameID, index, key->depth, before, aft, key->mask, kp1,kp2);


    double norm2 = calP(result, key->depth, after_pt, before, aft, key->mask, after, kp1,kp2);


/*
    left_ds = before.clone();
    right_ds = aft.clone();


    for(int i = 0; i < kp1.size(); i++){
        circle(left_ds, Point2f(kp1.at(i).pt.x, kp1.at(i).pt.y), 5, Scalar(0,0,255),-1);
        circle(right_ds, Point2f(kp2.at(i).pt.x, kp2.at(i).pt.y), 5, Scalar(0,0,255),-1);

    }
        imshow("left", right_ds);
        imshow("right", left_ds);
        waitKey(0);
*/

    double norm = normofTransform(result.rvec, result.tvec);

    cout << "norm: " << norm << " norm2: " << norm2 << endl;

    FRAME* key_x = new FRAME();
    key_x->mask = after.clone();
    key_x->rgb = img_left.clone();

        Vec3b p;
        p[0] = 0;
        p[1] = 0;
        p[2] = 0;
        for(int i = 0; i < img_left.rows; i++)
            for(int j = left_c.cols/2; j < left_c.cols; j++)
                key_x->rgb.at<Vec3b>(i,j) = p;


    key_x->disparity = disp.clone();
    key_x->frameID = index;
    key_x->depth.insert(after_pt.begin(), after_pt.end());
    after_pt.clear();

    if(norm > 1 || norm2 > 1.05 || norm2 < 0.95){
        cout << "-------------may be a bad frame------------" << endl;
        check_exist(globalOptimizer, key_x);
        //cout << "frame: " << index << endl;
        return;
    }

 
    if(norm < 0.001){

        return;
    }

    keyframes.push_back(key_x);
    cout << "add new vertex: " << key_x->frameID << endl;

    Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec );
    //g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( key->frameID ));
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId(index);
    v->setEstimate( Eigen::Isometry3d::Identity() );
    //v->setEstimate(vertex->estimate() * T.inverse());

    globalOptimizer.addVertex(v);
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    edge->setVertex( 0, globalOptimizer.vertex(key->frameID ));
    edge->setVertex(1, globalOptimizer.vertex( key_x->frameID) );
    //edge->setRobustKernel( robustKernel  );

    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
    information(0,0) = information(1,1) = information(2,2) = 1000.0;
    information(3,3) = information(4,4) = information(5,5) = 1000.0;


    edge->setInformation( information );

    edge->setMeasurement( T.inverse() );

    globalOptimizer.addEdge(edge);


    if(check_loop){
        checkNearbyLoops(globalOptimizer, key_x);
        checkRandomLoops(globalOptimizer, key_x);
    }
}




PointCloud::Ptr image2PointCloud(FRAME key){
    cv::Mat pix = Mat::zeros(4, 1, CV_64FC1);  
    cv::Mat pixW = Mat::zeros(4, 1, CV_64FC1);  

	PointCloud::Ptr cloud2 ( new PointCloud );

    cv::Mat Q = Mat::zeros(4, 4, CV_64FC1); 
    Q.at<double>(0,0) = 1;
    Q.at<double>(0, 3) = -left_c.cx;
    Q.at<double>(1,1) = 1;
    Q.at<double>(1,3) = -left_c.cy;
    Q.at<double>(2,3) = (left_c.fx+left_c.fy)/2;
    Q.at<double>(3,2) = -1/left_c.baseline;  
    Q.at<double>(3,3) = left_c.disp / left_c.baseline;

    for(int m = 0; m < key.disparity.cols; m++)
        for(int n = 0; n < key.disparity.rows; n++){

            double d = key.disparity.at<double>(n,m);
	        if(abs(d) < 0.00001)continue;

	        pix.at<double>(0,0) = m;
    	    pix.at<double>(1,0) = n;
    	    pix.at<double>(2,0) = d;
    	    pix.at<double>(3,0) = 1;
        
	        pixW = Q * pix;

        	PointT p;
       	 	p.z = -pixW.at<double>(2,0)/pixW.at<double>(3,0);
        	p.x = -pixW.at<double>(0,0)/pixW.at<double>(3,0);
        	p.y = -pixW.at<double>(1,0)/pixW.at<double>(3,0);

            //cout << p.z << endl;


            Vec3b color = key.rgb.at<Vec3b>(n,m);
            p.b = color[0];
            p.g = color[1];
            p.r = color[2];

            //if(p.b < 10 && p.g < 10 && p.r < 10)continue;

            cloud2->points.push_back(p);
		}

		cloud2->height = 1;
	    cloud2->width = cloud2->points.size();
	    cloud2->is_dense = false;
		return cloud2;
}


PointCloud::Ptr noColorPointCloud(FRAME key){
    cv::Mat pix = Mat::zeros(4, 1, CV_64FC1);  
    cv::Mat pixW = Mat::zeros(4, 1, CV_64FC1);  

    PointCloud::Ptr cloud2 ( new PointCloud );

    cv::Mat Q = Mat::zeros(4, 4, CV_64FC1); 
    Q.at<double>(0,0) = 1;
    Q.at<double>(0, 3) = -left_c.cx;
    Q.at<double>(1,1) = 1;
    Q.at<double>(1,3) = -left_c.cy;
    Q.at<double>(2,3) = (left_c.fx+left_c.fy)/2;
    Q.at<double>(3,2) = -1/left_c.baseline;  
    Q.at<double>(3,3) = left_c.disp / left_c.baseline;

    for(int m = 0; m < key.disparity.cols; m++)
        for(int n = 0; n < key.disparity.rows; n++){

            double d = key.disparity.at<double>(n,m);
            if(abs(d) < 0.00001)continue;

            pix.at<double>(0,0) = m;
            pix.at<double>(1,0) = n;
            pix.at<double>(2,0) = d;
            pix.at<double>(3,0) = 1;
        
            pixW = Q * pix;

            PointT p;
            p.z = -pixW.at<double>(2,0)/pixW.at<double>(3,0);
            p.x = -pixW.at<double>(0,0)/pixW.at<double>(3,0);
            p.y = -pixW.at<double>(1,0)/pixW.at<double>(3,0);

            //cout << p.z << endl;

            p.b = 0;
            p.g = 0;
            p.r = 0;

            cloud2->points.push_back(p);
        }

        cloud2->height = 1;
        cloud2->width = cloud2->points.size();
        cloud2->is_dense = false;
        return cloud2;
}

int main( int argc, char** argv ){
    //init(1000000);



    g2o::SparseOptimizer globalOptimizer;

        left_c.fx = 1297.957134;
        left_c.fy = 1297.749061;
        left_c.cx = 635.264100;
        left_c.cy = 547.933889;
        left_c.baseline = 782.532175/1425.111087;
        left_c.disp = 635.264100 - 580.347172;

        left_c.cols = 1224;
        left_c.rows = 1024;
  cin >> pic_start >> pic_cnt;
  
  vector<string> left_names;
  vector<string> right_names;
  get_file_names(left_names,right_names);


  // 初始化求解器
  SlamLinearSolver* linearSolver = new SlamLinearSolver();
  linearSolver->setBlockOrdering( false );
  SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );
    // 最后用的就是这个东东
  globalOptimizer.setAlgorithm( solver ); 
  // 不要输出调试信息
  globalOptimizer.setVerbose( true );

  g2o::VertexSE3* v = new g2o::VertexSE3();
  v->setId(0);
  cout << "add new vertex: " << 0 << endl;
  v->setEstimate( Eigen::Isometry3d::Identity() ); 
  v->setFixed( true ); 
  globalOptimizer.addVertex( v );

  for(int index = 0; index < left_names.size(); index++){

    get_depth(globalOptimizer, index, left_names[index], right_names[index]);
    int inde = rand()%4;
    
    if(inde)check_loop = true;
    else
            check_loop = false;

  }
  //sort_feat();
  globalOptimizer.verifyInformationMatrices();
  cout<<"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
  globalOptimizer.save("./data/result_before.g2o");
  globalOptimizer.setVerbose(true);
  globalOptimizer.initializeOptimization();
  globalOptimizer.optimize( 1000 ); //可以指定优化步数
  globalOptimizer.save( "./data/result_after.g2o" );
  cout<<"Optimization done."<<endl;

  PointCloud::Ptr output ( new PointCloud() ); //全局地图
  PointCloud::Ptr tmp ( new PointCloud() );

  pcl::VoxelGrid<PointT> voxel;
  pcl::PassThrough<PointT> pass;
  pass.setFilterFieldName("z");
  pass.setFilterLimits( 0, 40.0 ); 
  double gridsize = 0.001; 
  voxel.setLeafSize( gridsize, gridsize, gridsize );

   get_color(globalOptimizer);

  for (size_t i=0; i<keyframes.size(); i++){
      g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[i]->frameID ));
      Eigen::Isometry3d pose = vertex->estimate();
      
      cout << endl << pose.matrix() << endl;


      string name = "./data/pointcloud";

      PointCloud::Ptr newCloud = image2PointCloud(*keyframes[i]); //转成点云

      // 以下是滤波
      voxel.setInputCloud( newCloud );
      voxel.filter( *tmp );
      pass.setInputCloud( tmp );
      pass.filter( *newCloud );
        // 把点云变换后加入全局地图中
      pcl::transformPointCloud( *newCloud, *tmp, pose.matrix() );
      *output += *tmp;

      tmp->clear();
      newCloud->clear();

  }

  voxel.setInputCloud( output );
  voxel.filter( *tmp );
  //存储
  pcl::io::savePCDFile( "./data/result.pcd", *tmp );
  pcl::io::savePLYFile( "./data/pointcloud.ply", *tmp );  
  cout<<"Final map is saved."<<endl;
  tmp->points.clear();
  output->points.clear();
  globalOptimizer.clear();


  return 0;
}

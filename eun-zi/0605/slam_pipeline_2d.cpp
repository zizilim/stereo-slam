#include <opencv2/opencv.hpp> 
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <deque>
#include <thread>

#define ORBTHRESHOLD 10
#define HORIZENTALITY 40
#define MATCHTHRESHOLD 40

using namespace std;
using namespace cv;

string get_pipeline(int sensor_id, int width, int height, int fps) {
    return "nvarguscamerasrc sensor-id=" + to_string(sensor_id) +
           " exposuretimerange=\"20000000 20000000\" gainrange=\"128 128\" wbmode=1" +
           " ! video/x-raw(memory:NVMM), width=" + to_string(width) +
           ", height=" + to_string(height) + ", framerate=" + to_string(fps) + "/1 ! " +
           "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! " +
           "videoconvert ! video/x-raw, format=BGR ! appsink";
}

int main() {
    int width = 3264, height = 2464, fps = 21;
    VideoCapture cap0(get_pipeline(0, width, height, fps), cv::CAP_GSTREAMER);
    VideoCapture cap1(get_pipeline(1, width, height, fps), cv::CAP_GSTREAMER);

    if (!cap0.isOpened() || !cap1.isOpened()) {
        cerr << "[ERROR] 카메라 열기 실패!" << endl;
        return -1;
    }

    Mat map_img = Mat::zeros(Size(1280, 960), CV_8UC3);
    Mat img_left, img_right;

    Ptr<ORB> orb = ORB::create(10000, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, ORBTHRESHOLD);

    while (true) {
        cap0.read(img_left);
        cap1.read(img_right);
        if (img_left.empty() || img_right.empty()) continue;

        int key = waitKey(1);
        if (key == 27) break;      // ESC: 종료
        if (key != 'p') continue;  // 'p' 키 아니면 패스

        resize(img_left, img_left, Size(1280, 960));
        resize(img_right, img_right, Size(1280, 960));
        flip(img_left, img_left, -1);
        flip(img_right, img_right, -1);
        cvtColor(img_left, img_left, COLOR_BGR2GRAY);
        cvtColor(img_right, img_right, COLOR_BGR2GRAY);
        medianBlur(img_left, img_left, 3);
        medianBlur(img_right, img_right, 3);
        Ptr<CLAHE> clahe = createCLAHE(2.0, Size(4, 4));
        clahe->apply(img_left, img_left);
        clahe->apply(img_right, img_right);

        Mat canny_edges, canny_edges_right;
        vector<Vec4i> lines, lines_right;
        Canny(img_left, canny_edges, 50, 100);
        HoughLinesP(canny_edges, lines, 1, CV_PI / 180, 40, 10, 40);
        Canny(img_right, canny_edges_right, 50, 100);
        HoughLinesP(canny_edges_right, lines_right, 1, CV_PI / 180, 40, 10, 40);

        Mat hough_mask = Mat::zeros(img_left.size(), CV_8UC1);
        for (const auto& l : lines)
            line(hough_mask, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 1);
        dilate(hough_mask, hough_mask, Mat(), Point(-1, -1), 2);

        Mat hough_mask_right = Mat::zeros(img_right.size(), CV_8UC1);
        for (const auto& l : lines_right)
            line(hough_mask_right, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 1);
        dilate(hough_mask_right, hough_mask_right, Mat(), Point(-1, -1), 2);

        vector<KeyPoint> kp_left, kp_right;
        Mat desc_left, desc_right;
        orb->detectAndCompute(img_left, hough_mask, kp_left, desc_left);
        orb->detectAndCompute(img_right, hough_mask_right, kp_right, desc_right);

        vector<DMatch> matches;
        BFMatcher matcher(NORM_HAMMING);
        matcher.match(desc_left, desc_right, matches);

        vector<DMatch> good_matches;
        for (const auto& m : matches) {
            Point2f pl = kp_left[m.queryIdx].pt;
            Point2f pr = kp_right[m.trainIdx].pt;
            float disparity = pr.x - pl.x;
            if (disparity > 1 && disparity < 100 && abs(pl.y - pr.y) < HORIZENTALITY) {
                good_matches.push_back(m);
            }
        }

        for (const auto& m : good_matches) {
            const KeyPoint& kp = kp_left[m.queryIdx];
            float response = kp.response;
            Scalar color = (response > 50) ? Scalar(0, 255, 0) : Scalar(0, 165, 255);
            circle(map_img, kp.pt, 2, color, -1);
        }

        imshow("2D Map", map_img);
        imwrite("/home/eunzi/Desktop/2d_map.jpg", map_img);
        cout << "[INFO] 'p' 키 입력 → 프레임 처리 완료" << endl;
    }

    cap0.release();
    cap1.release();
    destroyAllWindows();
    return 0;
}

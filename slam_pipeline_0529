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

std::string get_pipeline(int sensor_id, int width, int height, int fps) {
    return "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) +
           " exposuretimerange=\"20000000 20000000\" gainrange=\"128 128\" wbmode=1" +
           " ! video/x-raw(memory:NVMM), width=" + std::to_string(width) +
           ", height=" + std::to_string(height) + ", framerate=" + std::to_string(fps) + "/1 ! " +
           "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! " +
           "videoconvert ! video/x-raw, format=BGR ! appsink";
}


int main() {
    // 좌우 이미지 불러오기 (그레이스케일)
    int width = 3264, height = 2464, fps = 21;

    VideoCapture cap0(get_pipeline(0, width, height, fps), cv::CAP_GSTREAMER);
    VideoCapture cap1(get_pipeline(1, width, height, fps), cv::CAP_GSTREAMER);

    if (!cap0.isOpened() || !cap1.isOpened()) {
        cerr << "카메라 열기 실패!" << endl;
        return -1;
    }

    Mat img_left;
    Mat img_right;
    
    int wait_count = 0;
    while(wait_count < 50){
        cap0.read(img_left);
        cap1.read(img_right);
        if(!img_left.empty() && !img_right.empty()){
            break;
        }
        wait_count++;
        this_thread::sleep_for(chrono::milliseconds(20));
    }

    if(img_left.empty() || img_right.empty()){
        cout << "Cannot read Image" << endl;
        return -1;
    }

    resize(img_left, img_left, Size(1280, 960));
    resize(img_right, img_right, Size(1280, 960));

    flip(img_left, img_left, -1);
    flip(img_right, img_right, -1);
    
    imwrite("leftImage.jpg", img_left);
    imwrite("rightImage.jpg", img_right);

    cvtColor(img_left, img_left, COLOR_BGR2GRAY);
    cvtColor(img_right, img_right, COLOR_BGR2GRAY);
    
    medianBlur(img_left, img_left, 3);
    medianBlur(img_right, img_right, 3);

    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(4, 4));
    clahe->apply(img_left, img_left);
    clahe->apply(img_right, img_right);

    // 2. ORB 특징점 검출기 생성
    Ptr<ORB> orb = ORB::create(200000, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, ORBTHRESHOLD);  // 최대 1000개의 키포인트

    // Canny 수행
    Mat canny_edges, canny_edges_right;
    vector<Vec4i> lines, lines_right;

    Canny(img_left, canny_edges, 50, 100);
    HoughLinesP(canny_edges, lines, 1, CV_PI / 180, 40, 10, 40);

    Canny(img_right, canny_edges_right, 50, 100);
    HoughLinesP(canny_edges_right, lines_right, 1, CV_PI / 180, 40, 10, 40);

    Mat hough_mask = Mat::zeros(img_left.size(), CV_8UC1);
    for (const auto& hline : lines) {
        line(hough_mask, Point(hline[0], hline[1]), Point(hline[2], hline[3]), Scalar(255), 1);
    }
    dilate(hough_mask, hough_mask, Mat(), Point(-1, -1), 2);

    Mat hough_mask_right = Mat::zeros(img_right.size(), CV_8UC1);
    for (const auto& hline : lines_right) {
        line(hough_mask_right, Point(hline[0], hline[1]), Point(hline[2], hline[3]), Scalar(255), 1);
    }
    dilate(hough_mask_right, hough_mask_right, Mat(), Point(-1, -1), 2);

    // ORB 특징점 추출
    vector<KeyPoint> kp_left_raw, kp_right_raw;
    Mat desc_left_raw, desc_right_raw;
    orb->detectAndCompute(img_left, hough_mask, kp_left_raw, desc_left_raw);
    orb->detectAndCompute(img_right, hough_mask_right, kp_right_raw, desc_right_raw);

    cout << "Canny-Orb keypoint number: " << kp_left_raw.size() << endl;

    // ORB 키포인트 필터링
    vector<KeyPoint> kp_left, kp_right;
    Mat desc_left, desc_right;

    vector<KeyPoint> filtered_kp;
    Mat filtered_desc;
    vector<int> sorted_indices;
    for(int i = 0; i < kp_left_raw.size(); i++){
        sorted_indices.push_back(i);
    }
    sort(sorted_indices.begin(), sorted_indices.end(),
            [&](int a, int b) {
                return kp_left_raw[a].response > kp_left_raw[b].response;
            });
    vector<bool> used(kp_left_raw.size(), false);

    const float CLUSTER_RADIUS = 10.0f;
    for(int i : sorted_indices){
        if(used[i]) continue;

        const Point2f& center = kp_left_raw[i].pt;
        filtered_kp.push_back(kp_left_raw[i]);
        filtered_desc.push_back(desc_left_raw.row(i));
        used[i] = true;

        for(int j = 0; j < kp_left_raw.size(); j++){
            if(!used[j] && norm(kp_left_raw[j].pt - center) < CLUSTER_RADIUS) {
                used[j] = true;
            }
        }
    }

    kp_left = filtered_kp;
    desc_left = filtered_desc.clone();

    sorted_indices.clear();
    filtered_kp.clear();
    filtered_desc.release();
    used.clear();

    //clusturing right image
    for(int i = 0; i < kp_right_raw.size(); i++){
        used.push_back(false);
        sorted_indices.push_back(i);
    }

    sort(sorted_indices.begin(), sorted_indices.end(),
            [&](int a, int b) {
                return kp_right_raw[a].response > kp_right_raw[b].response;
            });
            
    for(int i : sorted_indices){
        if(used[i]) continue;

        const Point2f& center = kp_right_raw[i].pt;
        filtered_kp.push_back(kp_right_raw[i]);
        filtered_desc.push_back(desc_right_raw.row(i));
        used[i] = true;

        for(int j = 0; j < kp_right_raw.size(); j++){
            if(!used[j] && norm(kp_right_raw[j].pt - center) < CLUSTER_RADIUS) {
                used[j] = true;
            }
        }
    }

    kp_right = filtered_kp;
    desc_right = filtered_desc.clone();

    // 특징점들을 y값을 기준으로 정렬
    vector<int> sortedLeft;
    for (int i = 0; i < kp_left.size(); i++) {
        sortedLeft.push_back(i);
    }

    sort(sortedLeft.begin(), sortedLeft.end(),
        [&](int a, int b) {
            return kp_left[a].pt.y < kp_left[b].pt.y;
        });

    vector<int> sortedRight;
    for (int i = 0; i < kp_right.size(); i++) {
        sortedRight.push_back(i);
    }

    sort(sortedRight.begin(), sortedRight.end(),
        [&](int a, int b) {
            return kp_right[a].pt.y < kp_right[b].pt.y;
        });

    
    // 에피폴라 라인 제약 매칭
    deque<int> trainIndex;
    int current = 0;

    std::vector<cv::DMatch> matches;
    vector<float> disparities;
    for (int i = 0; i < kp_left.size(); i++) {
        while (trainIndex.size() > 0 && kp_right[trainIndex[0]].pt.y < kp_left[sortedLeft[i]].pt.y - HORIZENTALITY) {
            trainIndex.erase(trainIndex.begin());
        }
        while (current < sortedRight.size() && kp_right[sortedRight[current]].pt.y <= kp_left[sortedLeft[i]].pt.y + HORIZENTALITY) {
            trainIndex.push_back(sortedRight[current]);
            current++;
        }

        int best_idx = -1;
        int best_dist = INT_MAX;
        for (int j = 0; j < trainIndex.size(); j++) {
            int dist = cv::norm(desc_left.row(sortedLeft[i]), desc_right.row(trainIndex[j]), NORM_HAMMING);
            if(kp_right[trainIndex[j]].pt.x - kp_left[sortedLeft[i]].pt.x < 0) continue;
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }

        if (best_dist < MATCHTHRESHOLD) {
            matches.emplace_back(sortedLeft[i], trainIndex[best_idx], static_cast<float>(best_dist));
            trainIndex.erase(trainIndex.begin() + best_idx);
        }
    }
    
    /*
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(desc_left, desc_right, matches);
    vector<float> disparities;
    */
    // 시차 계산 (X좌표 차이)
    for (const auto& match : matches) {
        Point2f pt_left = kp_left[match.queryIdx].pt;
        Point2f pt_right = kp_right[match.trainIdx].pt;
        float disparity = pt_right.x - pt_left.x;
        disparities.push_back(disparity);
        // 필요시: 깊이 = (f * baseline) / disparity;
    }
    
    for(auto n : disparities){
        cout << n << endl;
    }

    // ORB 매칭 시각화
    Mat match_img;
    drawMatches(img_left, kp_left, img_right, kp_right, matches, match_img,
        Scalar::all(-1), Scalar::all(-1),
        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // ORB 특징점만 시각화
    Mat orb_gradient_img;
    cvtColor(img_left, orb_gradient_img, COLOR_GRAY2BGR);

    // 이 이미지를 결과용으로 복사해서 라인 연결에 사용
    Mat direction_img;
    cvtColor(canny_edges, direction_img, COLOR_GRAY2BGR);

    Mat orb_img_left;
    cvtColor(img_left, orb_img_left, cv::COLOR_GRAY2BGR);

    for (const auto& l : lines) {
        
        Point2f A(l[0], l[1]);
        Point2f B(l[2], l[3]);
        /*
        bool has_near_kp_A = false;
        bool has_near_kp_B = false;

        for (const auto& kp : kp_left) {
            float dist_to_A = norm(kp.pt - A);
            float dist_to_B = norm(kp.pt - B);

            if (dist_to_A < 10) // 시작점이나 끝점 근처
                has_near_kp_A = true;
            if (dist_to_B < 10)
                has_near_kp_B = true;
            if(has_near_kp_A || has_near_kp_B)
                break;
        }
        */
        //if (has_near_kp_A || has_near_kp_B) {
            line(direction_img, A, B, Scalar(255, 0, 255), 2, LINE_AA);
        //}
    }

    for(int i = 0; i < kp_left.size(); i++){
        circle(direction_img, kp_left[i].pt, 3, Scalar(0, 255, 0), -1);
    }

    //img_right
    Mat direction_img_right;
    cvtColor(canny_edges_right, direction_img_right, COLOR_GRAY2BGR);
    for (const auto& l : lines_right) {        
        Point2f A(l[0], l[1]);
        Point2f B(l[2], l[3]);
        line(direction_img_right, A, B, Scalar(255, 0, 255), 2, LINE_AA);
    }

    for(int i = 0; i < kp_right.size(); i++){
        circle(direction_img_right, kp_right[i].pt, 3, Scalar(0, 255, 0), -1);
    }

    Mat orb_img_right;
    cvtColor(img_right, orb_img_right, cv::COLOR_GRAY2BGR);
    for(int i = 0; i < kp_right.size(); i++){
        circle(orb_img_right, kp_right[i].pt, 10, Scalar(0, 255, 0), -1);
    }

    Size newSize(640, 480);
    resize(match_img, match_img, Size(1280, 480));
    imshow("ORB Matches", match_img);

    resize(img_left, img_left, newSize);
    imshow("GrayScale left image", img_left);
    
    resize(img_right, img_right, newSize);
    imshow("GrayScale right image", img_right);
    
    resize(direction_img, direction_img, newSize);
    imshow("Canny-ORB with HoughLines", direction_img);

    resize(direction_img_right, direction_img_right, newSize);
    imshow("Canny-ORB with HoughLines_right", direction_img_right);

    cap0.release();
    cap1.release();

    waitKey(0);
    return 0;
}

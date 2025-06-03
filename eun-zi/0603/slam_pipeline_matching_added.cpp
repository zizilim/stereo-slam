#include <opencv2/opencv.hpp> 
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <deque>
#include <thread>

// ORB 추출 민감도, 수평 매칭 허용 범위, 매칭 임계값
#define ORBTHRESHOLD 10
#define HORIZENTALITY 40
#define MATCHTHRESHOLD 40

using namespace std;
using namespace cv;

// Jetson Nano의 CSI 카메라를 위한 GStreamer 파이프라인 설정 함수
std::string get_pipeline(int sensor_id, int width, int height, int fps) {
    return "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) +
           " exposuretimerange=\"20000000 20000000\" gainrange=\"128 128\" wbmode=1" +
           " ! video/x-raw(memory:NVMM), width=" + std::to_string(width) +
           ", height=" + std::to_string(height) + ", framerate=" + std::to_string(fps) + "/1 ! " +
           "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! " +
           "videoconvert ! video/x-raw, format=BGR ! appsink";
}

int main() {
    // 이미지 해상도 및 FPS 설정
    int width = 3264, height = 2464, fps = 21;

    // 카메라 열기
    VideoCapture cap0(get_pipeline(0, width, height, fps), cv::CAP_GSTREAMER);
    VideoCapture cap1(get_pipeline(1, width, height, fps), cv::CAP_GSTREAMER);

    if (!cap0.isOpened() || !cap1.isOpened()) {
        cerr << "\n[ERROR] 카메라 열기 실패!" << endl;
        return -1;
    }

    Mat img_left, img_right;

    // 카메라 준비 기다림 (최대 1초 대기)
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

    // 이미지 캡처 실패 시 종료
    if(img_left.empty() || img_right.empty()){
        cout << "Cannot read Image" << endl;
        return -1;
    }

    // 크기 조절 및 상하반전 (센서 방향 보정)
    resize(img_left, img_left, Size(1280, 960));
    resize(img_right, img_right, Size(1280, 960));
    flip(img_left, img_left, -1);
    flip(img_right, img_right, -1);

    // RGB 이미지 저장
    imwrite("leftImage.jpg", img_left);
    imwrite("rightImage.jpg", img_right);

    // 그레이 변환 + 잡음 제거 + 대비 보정
    cvtColor(img_left, img_left, COLOR_BGR2GRAY);
    cvtColor(img_right, img_right, COLOR_BGR2GRAY);
    medianBlur(img_left, img_left, 3);
    medianBlur(img_right, img_right, 3);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(4, 4));
    clahe->apply(img_left, img_left);
    clahe->apply(img_right, img_right);

    // ORB 객체 생성 (키포인트 최대 수, 스케일, 피라미드 수 등)
    Ptr<ORB> orb = ORB::create(1000, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, ORBTHRESHOLD);

    // Canny 엣지 + HoughLinesP 직선 검출
    Mat canny_edges, canny_edges_right;
    vector<Vec4i> lines, lines_right;
    Canny(img_left, canny_edges, 50, 100);
    HoughLinesP(canny_edges, lines, 1, CV_PI / 180, 40, 10, 40);
    Canny(img_right, canny_edges_right, 50, 100);
    HoughLinesP(canny_edges_right, lines_right, 1, CV_PI / 180, 40, 10, 40);

    // Hough 선을 마스크로 변환하여 ORB에 사용
    Mat hough_mask = Mat::zeros(img_left.size(), CV_8UC1);
    for (const auto& hline : lines)
        line(hough_mask, Point(hline[0], hline[1]), Point(hline[2], hline[3]), Scalar(255), 1);
    dilate(hough_mask, hough_mask, Mat(), Point(-1, -1), 2);

    Mat hough_mask_right = Mat::zeros(img_right.size(), CV_8UC1);
    for (const auto& hline : lines_right)
        line(hough_mask_right, Point(hline[0], hline[1]), Point(hline[2], hline[3]), Scalar(255), 1);
    dilate(hough_mask_right, hough_mask_right, Mat(), Point(-1, -1), 2);

    // ORB 키포인트 + 디스크립터 추출 (마스크 적용)
    vector<KeyPoint> kp_left_raw, kp_right_raw;
    Mat desc_left_raw, desc_right_raw;
    orb->detectAndCompute(img_left, hough_mask, kp_left_raw, desc_left_raw);
    orb->detectAndCompute(img_right, hough_mask_right, kp_right_raw, desc_right_raw);
    cout << "Canny-Orb keypoint number: " << kp_left_raw.size() << endl;

    // 키포인트 클러스터링 기반 필터링 (좌측)
    vector<KeyPoint> kp_left, kp_right, filtered_kp;
    Mat desc_left, desc_right, filtered_desc;
    vector<int> sorted_indices;
    for(int i = 0; i < kp_left_raw.size(); i++) sorted_indices.push_back(i);
    sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
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
            if(!used[j] && norm(kp_left_raw[j].pt - center) < CLUSTER_RADIUS) used[j] = true;
        }
    }
    kp_left = filtered_kp;
    desc_left = filtered_desc.clone();

    // 오른쪽 이미지 클러스터링 (동일한 방식)
    sorted_indices.clear(); filtered_kp.clear(); filtered_desc.release(); used.clear();
    for(int i = 0; i < kp_right_raw.size(); i++) {
        used.push_back(false);
        sorted_indices.push_back(i);
    }
    sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
        return kp_right_raw[a].response > kp_right_raw[b].response;
    });
    for(int i : sorted_indices){
        if(used[i]) continue;
        const Point2f& center = kp_right_raw[i].pt;
        filtered_kp.push_back(kp_right_raw[i]);
        filtered_desc.push_back(desc_right_raw.row(i));
        used[i] = true;
        for(int j = 0; j < kp_right_raw.size(); j++){
            if(!used[j] && norm(kp_right_raw[j].pt - center) < CLUSTER_RADIUS) used[j] = true;
        }
    }
    kp_right = filtered_kp;
    desc_right = filtered_desc.clone();

    // 수직 정렬 (y 좌표 기준) 후 에피폴라 제약 기반 매칭
    vector<int> sortedLeft, sortedRight;
    for (int i = 0; i < kp_left.size(); i++) sortedLeft.push_back(i);
    for (int i = 0; i < kp_right.size(); i++) sortedRight.push_back(i);
    sort(sortedLeft.begin(), sortedLeft.end(), [&](int a, int b) { return kp_left[a].pt.y < kp_left[b].pt.y; });
    sort(sortedRight.begin(), sortedRight.end(), [&](int a, int b) { return kp_right[a].pt.y < kp_right[b].pt.y; });

	std::vector<cv::DMatch> valid_matches;
    deque<int> trainIndex;
    int current = 0;
    vector<DMatch> matches;
    vector<float> disparities;
    for (int i = 0; i < kp_left.size(); i++) {
        while (!trainIndex.empty() && kp_right[trainIndex[0]].pt.y < kp_left[sortedLeft[i]].pt.y - HORIZENTALITY)
            trainIndex.pop_front();
        while (current < sortedRight.size() && kp_right[sortedRight[current]].pt.y <= kp_left[sortedLeft[i]].pt.y + HORIZENTALITY)
            trainIndex.push_back(sortedRight[current++]);

        int best_idx = -1, best_dist = INT_MAX;
        for (int j = 0; j < trainIndex.size(); j++) {
            int dist = norm(desc_left.row(sortedLeft[i]), desc_right.row(trainIndex[j]), NORM_HAMMING);
            if(kp_right[trainIndex[j]].pt.x - kp_left[sortedLeft[i]].pt.x < 0) continue;
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }
        if (best_dist < MATCHTHRESHOLD) {
            matches.emplace_back(sortedLeft[i], trainIndex[best_idx], (float)best_dist);
            trainIndex.erase(trainIndex.begin() + best_idx);
        }
    }

    // 시차 계산 (disparity = x 좌표 차이)
    for (const auto& match : matches) {
		Point2f pt_left = kp_left[match.queryIdx].pt;
		Point2f pt_right = kp_right[match.trainIdx].pt;
		float disparity = pt_right.x - pt_left.x;

		if (disparity > 0 && disparity < 100) {  // 유효 범위만 추가
			disparities.push_back(disparity);
			valid_matches.push_back(match);  // 유효 매칭만 따로 저장
		}
	}
    for(auto d : disparities) cout << d << endl;

    // 결과 시각화
    Mat match_img;
    drawMatches(img_left, kp_left, img_right, kp_right, valid_matches, match_img,
            Scalar::all(-1), Scalar::all(-1),
            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            
    imwrite("/home/eunzi/Desktop/orb_matches.jpg", match_img);

    Mat direction_img, direction_img_right;
    cvtColor(canny_edges, direction_img, COLOR_GRAY2BGR);
    cvtColor(canny_edges_right, direction_img_right, COLOR_GRAY2BGR);
    for (const auto& l : lines) line(direction_img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 255), 2);
    for (const auto& l : lines_right) line(direction_img_right, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 255), 2);
    for (const auto& kp : kp_left) circle(direction_img, kp.pt, 3, Scalar(0, 255, 0), -1);
    for (const auto& kp : kp_right) circle(direction_img_right, kp.pt, 3, Scalar(0, 255, 0), -1);

    resize(match_img, match_img, Size(1280, 480));
    resize(img_left, img_left, Size(640, 480));
    resize(img_right, img_right, Size(640, 480));
    resize(direction_img, direction_img, Size(640, 480));
    resize(direction_img_right, direction_img_right, Size(640, 480));

    imshow("ORB Matches", match_img);
    //imshow("GrayScale left image", img_left);
    //imshow("GrayScale right image", img_right);
    imshow("Canny-ORB with HoughLines", direction_img);
    imshow("Canny-ORB with HoughLines_right", direction_img_right);

    cap0.release();
    cap1.release();
    waitKey(0);
    return 0;
}

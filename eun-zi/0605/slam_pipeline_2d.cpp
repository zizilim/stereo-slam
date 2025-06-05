#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <deque>
#include <thread>
#include <cmath>
#include <chrono>

#define ORBTHRESHOLD 10
#define HORIZENTALITY 40
#define MATCHTHRESHOLD 40

using namespace std;
using namespace cv;
using namespace std::chrono;

// 회전 함수 추가: 방향 꼬깔 계산에 사용
Point2f rotate(const Point2f& vec, float angle_rad) {
    float cosA = cos(angle_rad);
    float sinA = sin(angle_rad);
    return Point2f(vec.x * cosA - vec.y * sinA, vec.x * sinA + vec.y * cosA);
}

// 카메라 파이프라인 문자열 생성 함수 (IMX219 Dual CSI 카메라용 GStreamer 파이프라인)
std::string get_pipeline(int sensor_id, int width, int height, int fps) {
    return "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) +
           " exposuretimerange=\"20000000 20000000\" gainrange=\"128 128\" wbmode=1" +
           " ! video/x-raw(memory:NVMM), width=" + std::to_string(width) +
           ", height=" + std::to_string(height) + ", framerate=" + std::to_string(fps) + "/1 ! " +
           "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! " +
           "videoconvert ! video/x-raw, format=BGR ! appsink";
}

int main() {
    // 카메라 해상도 및 FPS 설정
    int width = 3264, height = 2464, fps = 21;

    // VideoCapture 생성 (Dual IMX219)
    VideoCapture cap0(get_pipeline(0, width, height, fps), cv::CAP_GSTREAMER);
    VideoCapture cap1(get_pipeline(1, width, height, fps), cv::CAP_GSTREAMER);

    if (!cap0.isOpened() || !cap1.isOpened()) {
        cerr << "카메라 열기 실패!" << endl;
        return -1;
    }

    // 맵 초기화: 반복마다 다시 그릴 예정
    const int MAP_SIZE = 800;
    Point2f origin(MAP_SIZE / 2.0f, MAP_SIZE / 2.0f); // 맵 중앙
    float scale = 100.0f;                             // 1미터당 100픽셀

    // 스테레오 파라미터
    float fx = 500.0f;            // 초점 거리 (픽셀 단위)
    float baseline = 0.06f;       // 카메라 간 거리 6cm
    float cx = 640.0f / 2.0f;     // 영상 중심점 x
    float cy = 480.0f / 2.0f;     // 영상 중심점 y

    // 이전 프레임 위치 저장 변수
    Point2f prev_pos = origin;
    bool has_prev = false;

    // 높이 제한 (미터 단위): Y = (pt.y - cy) * Z / fx
    const float HEIGHT_LIMIT = 0.5f;  // ±0.5m 이내만 맵핑

    // 3초 간격 타이머 초기화
    auto last_time = steady_clock::now();

    while (true) {
        // 3초 주기 제어
        auto now = steady_clock::now();
        float elapsed = duration_cast<seconds>(now - last_time).count();
        if (elapsed < 3.0f) continue;
        last_time = now;

        // --- 새 맵 초기화 (매 반복마다 비움) ---
        Mat map = Mat::zeros(MAP_SIZE, MAP_SIZE, CV_8UC3);

        // 새 프레임 캡처
        Mat img_left, img_right;
        cap0.read(img_left);
        cap1.read(img_right);
        if (img_left.empty() || img_right.empty()) {
            cout << "카메라 이미지 없음!" << endl;
            break;
        }

        // 리사이즈 & 뒤집기
        resize(img_left, img_left, Size(1280, 960));
        resize(img_right, img_right, Size(1280, 960));
        flip(img_left, img_left, -1);
        flip(img_right, img_right, -1);

        // 그레이스케일 변환 & 노이즈 제거
        cvtColor(img_left, img_left, COLOR_BGR2GRAY);
        cvtColor(img_right, img_right, COLOR_BGR2GRAY);
        medianBlur(img_left, img_left, 3);
        medianBlur(img_right, img_right, 3);

        // CLAHE 적용
        Ptr<CLAHE> clahe = createCLAHE(2.0, Size(4, 4));
        clahe->apply(img_left, img_left);
        clahe->apply(img_right, img_right);

        // ORB 생성
        Ptr<ORB> orb = ORB::create(
            200000,  // 최대 키포인트 수
            1.2f,    // 피라미드 스케일 팩터
            8,       // 피라미드 층 수
            31,      // 엣지 임계값
            0,       // 첫 번째 Pyramid 레벨
            2,       // 빨강 연속
            ORB::HARRIS_SCORE,
            31,
            ORBTHRESHOLD  // 응답 임계값
        );

        // Canny + HoughLines
        Mat canny_edges, canny_edges_right;
        vector<Vec4i> lines, lines_right;
        Canny(img_left, canny_edges, 50, 100);
        HoughLinesP(canny_edges, lines, 1, CV_PI / 180, 40, 10, 40);
        Canny(img_right, canny_edges_right, 50, 100);
        HoughLinesP(canny_edges_right, lines_right, 1, CV_PI / 180, 40, 10, 40);

        // Hough 마스크 생성 후 팽창
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

        // ORB 키포인트 필터링 (클러스터링)
        vector<KeyPoint> kp_left, kp_right;
        Mat desc_left, desc_right;

        // 왼쪽 클러스터링
        vector<int> sorted_indices;
        sorted_indices.reserve(kp_left_raw.size());
        for (int i = 0; i < (int)kp_left_raw.size(); i++) sorted_indices.push_back(i);
        sort(sorted_indices.begin(), sorted_indices.end(),
             [&](int a, int b) { return kp_left_raw[a].response > kp_left_raw[b].response; });

        vector<bool> used(kp_left_raw.size(), false);
        const float CLUSTER_RADIUS = 10.0f;
        vector<KeyPoint> filtered_kp;
        Mat filtered_desc;
        for (int idx : sorted_indices) {
            if (used[idx]) continue;
            Point2f center = kp_left_raw[idx].pt;
            filtered_kp.push_back(kp_left_raw[idx]);
            filtered_desc.push_back(desc_left_raw.row(idx));
            used[idx] = true;
            for (int j = 0; j < (int)kp_left_raw.size(); j++) {
                if (!used[j] && norm(kp_left_raw[j].pt - center) < CLUSTER_RADIUS) {
                    used[j] = true;
                }
            }
        }
        kp_left = filtered_kp;
        desc_left = filtered_desc.clone();

        // 오른쪽 클러스터링
        sorted_indices.clear();
        used.clear();
        sorted_indices.reserve(kp_right_raw.size());
        for (int i = 0; i < (int)kp_right_raw.size(); i++) {
            sorted_indices.push_back(i);
            used.push_back(false);
        }
        sort(sorted_indices.begin(), sorted_indices.end(),
             [&](int a, int b) { return kp_right_raw[a].response > kp_right_raw[b].response; });

        filtered_kp.clear();
        filtered_desc.release();
        for (int idx : sorted_indices) {
            if (used[idx]) continue;
            Point2f center = kp_right_raw[idx].pt;
            filtered_kp.push_back(kp_right_raw[idx]);
            filtered_desc.push_back(desc_right_raw.row(idx));
            used[idx] = true;
            for (int j = 0; j < (int)kp_right_raw.size(); j++) {
                if (!used[j] && norm(kp_right_raw[j].pt - center) < CLUSTER_RADIUS) {
                    used[j] = true;
                }
            }
        }
        kp_right = filtered_kp;
        desc_right = filtered_desc.clone();

        // 특징점 Y값 기준 정렬
        vector<int> sortedLeft, sortedRight;
        sortedLeft.reserve(kp_left.size());
        sortedRight.reserve(kp_right.size());
        for (int i = 0; i < (int)kp_left.size(); i++) sortedLeft.push_back(i);
        for (int i = 0; i < (int)kp_right.size(); i++) sortedRight.push_back(i);
        sort(sortedLeft.begin(), sortedLeft.end(),
             [&](int a, int b) { return kp_left[a].pt.y < kp_left[b].pt.y; });
        sort(sortedRight.begin(), sortedRight.end(),
             [&](int a, int b) { return kp_right[a].pt.y < kp_right[b].pt.y; });

        // 에피폴라 라인 제약 매칭
        deque<int> trainIndex;
        int current = 0;
        vector<DMatch> matches;
        matches.reserve(min(kp_left.size(), kp_right.size()));
        for (int i = 0; i < (int)kp_left.size(); i++) {
            int li = sortedLeft[i];
            while (!trainIndex.empty() && kp_right[trainIndex.front()].pt.y < kp_left[li].pt.y - HORIZENTALITY) {
                trainIndex.pop_front();
            }
            while (current < (int)sortedRight.size() && kp_right[sortedRight[current]].pt.y <= kp_left[li].pt.y + HORIZENTALITY) {
                trainIndex.push_back(sortedRight[current]);
                current++;
            }
            int best_idx = -1;
            int best_dist = INT_MAX;
            for (int j = 0; j < (int)trainIndex.size(); j++) {
                int ri = trainIndex[j];
                if (kp_right[ri].pt.x - kp_left[li].pt.x < 0) continue;
                int dist = norm(desc_left.row(li), desc_right.row(ri), NORM_HAMMING);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = j;
                }
            }
            if (best_idx >= 0 && best_dist < MATCHTHRESHOLD) {
                int ri = trainIndex[best_idx];
                matches.emplace_back(li, ri, (float)best_dist);
                trainIndex.erase(trainIndex.begin() + best_idx);
            }
        }

        // 시차 계산 및 2D 맵핑 (높이 제한 포함)
        float sumX = 0, sumZ = 0;
        int count = 0;
        for (auto& match : matches) {
            Point2f ptL = kp_left[match.queryIdx].pt;
            Point2f ptR = kp_right[match.trainIdx].pt;
            float disp = ptR.x - ptL.x;
            if (disp < 0.5f) continue;  // 임계값 완화

            float Z = fx * baseline / disp;
            float Y = (ptL.y - cy) * Z / fx;          // 높이값 계산
            if (fabs(Y) > HEIGHT_LIMIT) continue;      // 높이 범위 밖이면 제외

            float X = (ptL.x - cx) * Z / fx;           // 좌우 위치 (미터 단위 환산)
            sumX += X;
            sumZ += Z;
            count++;

            // 2D 맵 좌표로 변환
            Point map_pt;
            map_pt.x = origin.x + static_cast<int>(X * scale);
            map_pt.y = origin.y - static_cast<int>(Z * scale);
            if (map_pt.inside(Rect(0, 0, MAP_SIZE, MAP_SIZE))) {
                circle(map, map_pt, 2, Scalar(255, 255, 255), -1); // 흰 점 = 맵 상 포인트
            }
        }

        // 현재 위치 및 방향 표시
        if (count > 0) {
            float meanX = sumX / count;
            float meanZ = sumZ / count;
            Point2f curr_pos = origin + Point2f(meanX * scale, -meanZ * scale);

            // === 방향 꼬깔 표시 (파랑) ===
            if (has_prev) {
                Point2f dir = curr_pos - prev_pos;
                if (norm(dir) > 1e-3) {
                    dir *= (1.0f / norm(dir));
                    Point2f left = curr_pos + rotate(dir, CV_PI / 6.0f) * 20.0f;
                    Point2f right = curr_pos + rotate(dir, -CV_PI / 6.0f) * 20.0f;
                    vector<Point> cone = { curr_pos, left, right };
                    fillConvexPoly(map, cone, Scalar(255, 0, 0));
                }
            }

            // === 현재 위치 점 표시 (노란색) ===
            circle(map, curr_pos, 5, Scalar(0, 255, 255), -1);

            prev_pos = curr_pos;
            has_prev = true;
        }

        // 매칭 결과 시각화
        Mat match_img;
        drawMatches(img_left, kp_left, img_right, kp_right, matches, match_img,
                    Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Canny+Hough 라인 + 키포인트 시각화 (선택)
        Mat direction_img;
        cvtColor(canny_edges, direction_img, COLOR_GRAY2BGR);
        for (auto& l : lines) {
            Point2f A(l[0], l[1]), B(l[2], l[3]);
            line(direction_img, A, B, Scalar(255, 0, 255), 2, LINE_AA);
        }
        for (auto& kp : kp_left) {
            circle(direction_img, kp.pt, 3, Scalar(0, 255, 0), -1);
        }

        Mat direction_img_right;
        cvtColor(canny_edges_right, direction_img_right, COLOR_GRAY2BGR);
        for (auto& l : lines_right) {
            Point2f A(l[0], l[1]), B(l[2], l[3]);
            line(direction_img_right, A, B, Scalar(255, 0, 255), 2, LINE_AA);
        }
        for (auto& kp : kp_right) {
            circle(direction_img_right, kp.pt, 3, Scalar(0, 255, 0), -1);
        }

        Mat orb_img_right;
        cvtColor(img_right, orb_img_right, COLOR_GRAY2BGR);
        for (auto& kp : kp_right) {
            circle(orb_img_right, kp.pt, 10, Scalar(0, 255, 0), -1);
        }

        // 창 크기 재조정 및 표시
        Size newSize(640, 480);
        resize(match_img, match_img, Size(1280, 480));
        imshow("ORB Matches", match_img);

        resize(img_left, img_left, newSize);
        //imshow("GrayScale left image", img_left);

        resize(img_right, img_right, newSize);
        //imshow("GrayScale right image", img_right);

        resize(direction_img, direction_img, newSize);
        //imshow("Canny-ORB with HoughLines", direction_img);

        resize(direction_img_right, direction_img_right, newSize);
        //imshow("Canny-ORB with HoughLines_right", direction_img_right);

        // 맵 보여주기 (현재 위치 점이 꼬깔 위에 그려짐)
        imshow("2D Mapping", map);

        // ESC 키를 누르면 종료
        if (waitKey(30) == 27) break;
    }

    cap0.release();
    cap1.release();
    destroyAllWindows();
    return 0;
}

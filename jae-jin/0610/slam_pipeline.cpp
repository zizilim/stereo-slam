#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <deque>
#include <thread>
#include <cmath>

#define _USE_MATH_DEFINES

#define ORBTHRESHOLD 10
#define HORIZENTALITY 20
#define MATCHTHRESHOLD 40
//1px = 1cm
#define PIXELSIZE 1
#define THICKNESS 2
#define BASELINE 80

using namespace std;
using namespace cv;

bool isPointOnLine(Point2f p, Vec4f line) {
    cv::Point2f a(line[0], line[1]);
    cv::Point2f b(line[2], line[3]);

    cv::Point2f ap = p - a;
    cv::Point2f ab = b - a;
    double ab_norm = sqrt(ab.x * ab.x + ab.y * ab.y);
    double cross = std::abs(ap.x * ab.y - ap.y * ab.x);
    if (cross / ab_norm > THICKNESS)
        return false;

    // 2. 선분 범위 내에 있는지 확인 (내적)
    double dot = ap.x * ab.x + ap.y * ab.y;
    if (dot < -THICKNESS || dot > ab.x * ab.x + ab.y * ab.y + THICKNESS)
        return false;

    return true;
}

bool isOnMap(int x, int y) {
    if (x >= 0 && x < 1000 && y >= 0 && y < 1000)
        return true;
    return false;
}

std::string get_pipeline(int sensor_id, int width, int height, int fps) {
    return "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) +
        " aelock=true awblock=true" +
        " exposuretimerange=\"16666667 16666667\" gainrange=\"4 4\" wbmode=1" +
        " ! video/x-raw(memory:NVMM), width=" + std::to_string(width) +
        ", height=" + std::to_string(height) + ", framerate=" + std::to_string(fps) + "/1 ! " +
        "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! " +
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false";
}


int main() {
    const float fx = 1333.0f;

    int width = 3264, height = 2464, fps = 15;
    Size imageSize = Size(1632, 1232);

    VideoCapture cap0(get_pipeline(0, width, height, fps), cv::CAP_GSTREAMER);
    VideoCapture cap1(get_pipeline(1, width, height, fps), cv::CAP_GSTREAMER);

    if (!cap0.isOpened() || !cap1.isOpened()) {
        cerr << "카메라 열기 실패!" << endl;
        return -1;
    }

    //맵 생성 (10m x 10m)
    Mat map = Mat::zeros(1000, 1000, CV_8UC1);
    int self_x = 500, self_z = 500;
    double radian = 0.0;

    //맵 표시
    Mat map_visual;
    cvtColor(map, map_visual, COLOR_GRAY2BGR);
    if (0 <= self_x && self_x < map.cols && 0 <= self_z && self_z < map.rows)
        circle(map_visual, Point(self_x, self_z), 5, Scalar(0, 255, 0), -1);

    imshow("STEREO VISION V-SLAM Map", map_visual);

    // 이전 프레임 정보
    vector<pair<double, double> > point_prev;
    Mat desc_prev;

    // 작동 시작
    while (true) {
        char key = (char)waitKey(0);
        if (key == 27) break; // esc 누르면 종료

        // space 누르면 촬영 및 매핑 (촬영과 매핑 코드 분리 필요)
        if (key == ' ') {
            cout << "Process is running..." << endl;
            Mat img_left = Mat::zeros(imageSize, CV_32FC1);
            Mat img_right = Mat::zeros(imageSize, CV_32FC1);
            
            for(int i = 0; i < 6; i++){
                Mat capture_left, capture_right;

                cap0.read(capture_left);
                cap1.read(capture_right);

                resize(capture_left, capture_left, imageSize);
                resize(capture_right, capture_right, imageSize);

                Mat fimg_left, fimg_right;
                cvtColor(capture_left, capture_left, COLOR_BGR2GRAY);
                capture_left.convertTo(fimg_left, CV_32FC1);
                img_left += fimg_left;
                cvtColor(capture_right, capture_right, COLOR_BGR2GRAY);
                capture_right.convertTo(fimg_right, CV_32FC1);
                img_right += fimg_right;
            }
            img_left /= 6.0;
            img_right /= 6.0;

            img_left.convertTo(img_left, CV_8UC1);
            img_right.convertTo(img_right, CV_8UC1);

            if (img_left.empty() || img_right.empty()) {
                cout << "Cannot read Image" << endl;
                return -1;
            }

            flip(img_left, img_left, -1);
            flip(img_right, img_right, -1);

            /*
            // Remap 적용
            Mat rect_left, rect_right;
            remap(img_left, rect_left, map1L, map2L, cv::INTER_LINEAR);
            remap(img_right, rect_right, map1R, map2R, cv::INTER_LINEAR);

            img_left = rect_left;
            img_right = rect_right;
            */


            // 이미지 후처리
            //cvtColor(img_left, img_left, COLOR_BGR2GRAY);
            //cvtColor(img_right, img_right, COLOR_BGR2GRAY);
            

            Ptr<CLAHE> clahe = createCLAHE(1.0, Size(16, 16));
            clahe->apply(img_left, img_left);
            clahe->apply(img_right, img_right);
            

            Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(LSD_REFINE_STD);
            vector<Vec4f> lines, lines_right;

            lsd->detect(img_left, lines);
            lsd->detect(img_right, lines_right);

            /*
            // Canny 수행
            Mat canny_edges, canny_edges_right;
            vector<Vec4i> lines, lines_right;

            Canny(img_left, canny_edges, 50, 100);
            HoughLinesP(canny_edges, lines, 1, CV_PI / 180, 40, 10, 40);

            Canny(img_right, canny_edges_right, 50, 100);
            HoughLinesP(canny_edges_right, lines_right, 1, CV_PI / 180, 40, 10, 40);
            */

            Mat hough_mask = Mat::zeros(img_left.size(), CV_8UC1);
            for (const auto& hline : lines) {
                line(hough_mask, Point(hline[0], hline[1]), Point(hline[2], hline[3]), Scalar(255), 3);
            }

            Mat hough_mask_right = Mat::zeros(img_right.size(), CV_8UC1);
            for (const auto& hline : lines_right) {
                line(hough_mask_right, Point(hline[0], hline[1]), Point(hline[2], hline[3]), Scalar(255), 3);
            }
            
            // ORB 특징점 추출
            Ptr<ORB> orb = ORB::create(2000, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, ORBTHRESHOLD);
            vector<KeyPoint> kp_left, kp_right;
            Mat desc_left, desc_right;
            
            orb->detectAndCompute(img_left, hough_mask, kp_left, desc_left);
            orb->detectAndCompute(img_right, hough_mask_right, kp_right, desc_right);
    
            int center = img_left.cols / 2;
            int center_y = img_left.rows / 2;

            BFMatcher matcher(NORM_HAMMING, false);
            vector<DMatch> matches;
            Mat desc_matched;
            matcher.match(desc_left, desc_right, matches);

            cout << kp_left.size() << ' ' << kp_right.size() << endl;
            cout << matches.size() << endl;
            vector<pair<KeyPoint, int> > kp_matched;
            vector<DMatch> good_matches;
            for (const auto& m : matches) {
                if (m.distance < 80) {  // 임계값 적용
                    int dist = kp_right[m.trainIdx].pt.y - kp_left[m.queryIdx].pt.y;
                    if (dist < -HORIZENTALITY || dist > HORIZENTALITY) continue;

                    double disparity = kp_right[m.trainIdx].pt.x - kp_left[m.queryIdx].pt.x;
                    if (disparity <= 0) continue;
                    double depth = 25000 / disparity;
                    if (depth > 10 && depth < 500) {
                        if ((kp_left[m.queryIdx].pt.y - center_y) * depth / 1550 < -100) continue;
                        kp_matched.push_back(make_pair(kp_left[m.queryIdx], depth));
                        cout << kp_left[m.queryIdx].pt << ' ' << kp_right[m.trainIdx].pt << endl;
                        desc_matched.push_back(desc_left.row(m.queryIdx));
                        good_matches.push_back(m);
                    }
                }
            }
            matches = good_matches;
            good_matches.clear();
            cout << matches.size() << endl;


            /*
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
                    if (kp_right[trainIndex[j]].pt.x - kp_left[sortedLeft[i]].pt.x < 20) continue;
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_idx = j;
                    }
                }

                if (best_dist < MATCHTHRESHOLD) {
                    matches.emplace_back(sortedLeft[i], trainIndex[best_idx], static_cast<float>(best_dist));
                    // 깊이 계산
                    Point2f pt_left = kp_left[sortedLeft[i]].pt;
                    Point2f pt_right = kp_right[trainIndex[best_idx]].pt;
                    double disparity = pt_right.x - pt_left.x;

                    double depth = 5000 / disparity;
                    if (depth > 20) {
                        desc_matched.push_back(desc_left.row(sortedLeft[i]));
                        kp_matched.push_back(make_pair(kp_left[sortedLeft[i]], depth));
                    }
                    trainIndex.erase(trainIndex.begin() + best_idx);
                }
            }
            */

            // 이전 프레임과 Brute-Force Matching
            vector<DMatch> matches_prev;
            if (desc_prev.rows != 0 && desc_matched.rows != 0)
                matcher.match(desc_prev, desc_matched, matches_prev);

            for (const auto& m : matches_prev) {
                if (m.distance < 20) {  // 임계값 적용
                    good_matches.push_back(m);
                }
            }
            matches_prev = good_matches;

            double rotate = 0.0;
            int sum = 0;

            // 각도 계산
            if (point_prev.size() != 0) {
                vector<int> leftPoints, rightPoints;
                for (auto l : lines) {
                    int left = -1, right = -1;
                    int left_prev = -1, right_prev = -1;
                    for (auto m : matches_prev) {
                        if (isPointOnLine(kp_matched[m.trainIdx].first.pt, l)) {
                            if (left == -1 || kp_matched[m.trainIdx].first.pt.x <= kp_matched[left].first.pt.x) {
                                left = m.trainIdx;
                                left_prev = m.queryIdx;
                            }

                            if (right == -1 || kp_matched[m.trainIdx].first.pt.x >= kp_matched[right].first.pt.x) {
                                right = m.trainIdx;
                                right_prev = m.queryIdx;
                            }
                        }
                    }
                    if (left != right) {
                        double theta1 = atan2(point_prev[right_prev].second - point_prev[left_prev].second, point_prev[right_prev].first - point_prev[left_prev].first);

                        double left_x = (kp_matched[left].first.pt.x - center) * kp_matched[left].second / fx / PIXELSIZE;
                        double right_x = (kp_matched[right].first.pt.x - center) * kp_matched[right].second / fx / PIXELSIZE;

                        double theta2 = atan2(kp_matched[right].second - kp_matched[left].second, right_x - left_x);
                        
                        int weight = 1;
                        if(isOnMap(point_prev[right_prev].second, point_prev[right_prev].first) && isOnMap(point_prev[left_prev].second, point_prev[left_prev].first))
                            weight = map.at<uchar>(point_prev[right_prev].second, point_prev[right_prev].first) + map.at<uchar>(point_prev[left_prev].second, point_prev[left_prev].first);
                        
                        rotate += (theta2 - theta1) * weight;
                        sum += weight;
                    }
                }
                if (sum != 0) {
                    radian += rotate / sum * 180.0 / 3.141592;
                    radian = fmod(radian + 2 * 3.141592, 2 * 3.141592);
                }
            }

            // 현재 위치 추정
            vector<pair<double, double> > point_matched;
            if (point_prev.size() != 0) {
                double nx = 0, nz = 0;
                int sum = 0;
                for (auto m : matches_prev) {
                    double x_prime = (kp_matched[m.trainIdx].first.pt.x - center) * kp_matched[m.trainIdx].second / fx / PIXELSIZE;
                    double z_prime = kp_matched[m.trainIdx].second / PIXELSIZE;

                    double x = x_prime * cos(radian) - z_prime * sin(radian);
                    double z = x_prime * sin(radian) + z_prime * cos(radian);

                    int weight = 1;
                    if(isOnMap(point_prev[m.queryIdx].second, point_prev[m.queryIdx].first))
                        weight = map.at<uchar>(point_prev[m.queryIdx].second, point_prev[m.queryIdx].first);
                    
                    nz += (point_prev[m.queryIdx].second + z) * weight;
                    nx += (point_prev[m.queryIdx].first - x) * weight;
                    sum += weight;
                }
                if (sum != 0) {
                    nx /= sum;
                    nz /= sum;

                    self_x = static_cast<int>(round(nx));
                    self_z = static_cast<int>(round(nz));
                }
            }

            // 프레임 상태 갱신
            point_prev.clear();
            desc_prev = Mat();

            for (int i = 0; i < kp_matched.size(); i++) {
                double x_prime = (kp_matched[i].first.pt.x - center) * kp_matched[i].second / fx / PIXELSIZE;
                double z_prime = kp_matched[i].second / PIXELSIZE;

                double x = x_prime * cos(radian) - z_prime * sin(radian);
                double z = x_prime * sin(radian) + z_prime * cos(radian);

                point_prev.push_back(make_pair(self_x + x, self_z - z));
                desc_prev.push_back(desc_matched.row(i).clone());
            }

            cout << "pos: " << self_x << ' ' << self_z << endl;
            cout << "rotate: " << radian << endl;

            // 같은 선분 상의 점 클러스터링
            for (auto l : lines) {
                int left = -1, right = -1;
                for (int i = 0; i < kp_matched.size(); i++) {
                    if (isPointOnLine(kp_matched[i].first.pt, l)) {
                        if (left == -1 || kp_matched[i].first.pt.x <= kp_matched[left].first.pt.x) {
                            left = i;
                        }

                        if (right == -1 || kp_matched[i].first.pt.x >= kp_matched[right].first.pt.x) {
                            right = i;
                        }
                    }
                }
                if (kp_matched[right].first.pt.x - kp_matched[left].first.pt.x < 10) continue;
                if (kp_matched[right].first.pt.x - kp_matched[left].first.pt.x >= 10) {
                    cout << kp_matched[left].first.pt << ' ' << kp_matched[right].first.pt << endl;
                }
                if (left == -1) continue;

                //선을 따라서 가중치 증가(매핑)
                Point p1(point_prev[left].first, point_prev[left].second);
                Point p2(point_prev[right].first, point_prev[right].second);
                cout << "point: " << p1 << ' ' << p2 << endl;
                cout << endl;
                
                LineIterator it(map, p1, p2);
                for (int i = 0; i < it.count; i++, ++it) {
                    Point pt = it.pos();
                    if (0 <= pt.y && pt.y < map.rows && 0 <= pt.x && pt.x < map.cols) {
                        if (map.at<uchar>(pt) < 245)
                            map.at<uchar>(pt) += 50;
                    }
                }
            }

            //맵 표시
            cvtColor(map, map_visual, COLOR_GRAY2BGR);
            if (0 <= self_x && self_x < map.cols && 0 <= self_z && self_z < map.rows)
                circle(map_visual, Point(self_x, self_z), 5, Scalar(0, 255, 0), -1);

            imshow("STEREO VISION V-SLAM Map", map_visual);


            // 이하 촬영 사진 출력
            // ORB 매칭 시각화
            Mat match_img;
            drawMatches(img_left, kp_left, img_right, kp_right, matches, match_img,
                Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            // ORB 특징점만 시각화
            Mat orb_gradient_img;
            cvtColor(img_left, orb_gradient_img, COLOR_GRAY2BGR);

            // 이 이미지를 결과용으로 복사해서 라인 연결에 사용
            Mat direction_img = orb_gradient_img.clone();
            //cvtColor(lines, direction_img, COLOR_GRAY2BGR);

            Mat orb_img_left;
            cvtColor(img_left, orb_img_left, cv::COLOR_GRAY2BGR);

            for (const auto& l : lines) {

                Point2f A(l[0], l[1]);
                Point2f B(l[2], l[3]);

                line(direction_img, A, B, Scalar(255, 0, 255), 2, LINE_AA);
            }

            for (int i = 0; i < kp_left.size(); i++) {
                circle(direction_img, kp_left[i].pt, 3, Scalar(0, 255, 0), -1);
            }

            //img_right
            Mat direction_img_right;
            cvtColor(img_right, direction_img_right, COLOR_GRAY2BGR);
            for (const auto& l : lines_right) {
                Point2f A(l[0], l[1]);
                Point2f B(l[2], l[3]);
                line(direction_img_right, A, B, Scalar(255, 0, 255), 2, LINE_AA);
            }

            for (int i = 0; i < kp_right.size(); i++) {
                circle(direction_img_right, kp_right[i].pt, 3, Scalar(0, 255, 0), -1);
            }

            Mat orb_img_right;
            cvtColor(img_right, orb_img_right, cv::COLOR_GRAY2BGR);
            for (int i = 0; i < kp_right.size(); i++) {
                circle(orb_img_right, kp_right[i].pt, 10, Scalar(0, 255, 0), -1);
            }

            Size newSize(640, 480);
            resize(match_img, match_img, Size(1280, 480));
            imshow("ORB Matches", match_img);

            resize(direction_img, direction_img, newSize);
            imshow("Canny-ORB with HoughLines", direction_img);

            resize(direction_img_right, direction_img_right, newSize);
            imshow("Canny-ORB with HoughLines_right", direction_img_right);
        }
    }

    return 0;
}

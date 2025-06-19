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
#define HORIZENTALITY 40
#define MATCHTHRESHOLD 40
#define LINEMATCHTHRESHOLD 1000
#define ERROR_THRESHOLD 20
//1px = 1cm
#define PIXELSIZE 1
#define THICKNESS 2
#define BASELINE 60
#define BASIC_WEIGHT 50

using namespace std;
using namespace cv;

const float fx = 1500.0f;
const Size imageSize = Size(1632, 1232);

bool isOnMap(int x, int y) {
    if (x >= 0 && x < 1000 && y >= 0 && y < 1000)
        return true;
    return false;
}


float getLineLength(Vec4f line) {
    float dx = line[2] - line[0];
    float dy = line[3] - line[1];
    return sqrt(dx * dx + dy * dy);
}

vector<Vec4f> mergeCollinearLines(vector<Vec4f> input_lines,
                                   float angle_thresh_deg = 5.0f,
                                   float point_dist_thresh = 1.0f,
                                   float vertical_reject_thresh = 1.0f) {
    vector<Vec4f> lines;

    for (const auto& l : input_lines) {
        Vec4f line = l;
        bool flip = false;
        if (line[0] > line[2]) flip = true;
        else if (line[0] == line[2] && line[1] > line[3]) flip = true;
        if (flip) {
            swap(line[0], line[2]);
            swap(line[1], line[3]);
        }

        if (abs(line[0] - line[2]) < vertical_reject_thresh)
            continue;  // 수직 또는 거의 수직 선분 제거

        lines.push_back(line);
    }

    sort(lines.begin(), lines.end(), [](const Vec4f& a, const Vec4f& b) {
        return a[0] < b[0];
    });

    vector<bool> used(lines.size(), false);
    vector<Vec4f> merged;

    for (size_t i = 0; i < lines.size(); ++i) {
        if (used[i]) continue;

        Vec4f base = lines[i];
        used[i] = true;

        Point2f start(base[0], base[1]);
        Point2f rightmost(base[2], base[3]);

        Point2f dir_base = rightmost - start;
        float norm_base = norm(dir_base);
        if (norm_base < 1e-6f) continue;
        Point2f dir_base_unit = dir_base / norm_base;

        for (size_t j = i + 1; j < lines.size(); ++j) {
            if (used[j]) continue;

            Vec4f cand = lines[j];
            Point2f cand_start(cand[0], cand[1]);

            if (cand_start.x > rightmost.x + 1.0f) break;

            Point2f dir_cand(cand[2] - cand[0], cand[3] - cand[1]);
            float norm_cand = norm(dir_cand);
            if (norm_cand < 1e-6f) continue;

            float dot = dir_base.dot(dir_cand) / (norm_base * norm_cand);
            if (dot < cos(angle_thresh_deg * CV_PI / 180.0f)) continue;

            Point2f ap = cand_start - start;
            float cross = fabs(dir_base.x * ap.y - dir_base.y * ap.x);
            float dist = cross / (norm_base + 1e-6f);
            if (dist > point_dist_thresh) continue;

            Point2f cand_end(cand[2], cand[3]);
            if (cand_end.x > rightmost.x)
                rightmost = cand_end;

            used[j] = true;
        }

        merged.emplace_back(Vec4f(start.x, start.y, rightmost.x, rightmost.y));
    }

    return merged;
}


bool isSameLine(Vec4f line1, Vec4f line2) {
    Point2f p1(line1[0], line1[1]);
    Point2f p2(line1[2], line1[3]);

    Point2f p3(line2[0], line2[1]);
    Point2f p4(line2[2], line2[3]);

    if (abs(p1.y - p3.y) > HORIZENTALITY || abs(p2.y - p4.y) > HORIZENTALITY)
        return false;
    
    float len_ratio = norm(p1 - p2) / norm(p3 - p4);
    if (len_ratio < 0.8 || len_ratio > 1.2)
        return false;

    float angle_diff = abs(atan2(p2.y - p1.y, p2.x - p1.x) - atan2(p4.y - p3.y, p4.x - p3.x)) * 180.0 / 3.141592;
    if (angle_diff > 10.0)
        return false;


    if(getLineLength(line1) > 600)
        cout << "Matched: " << line1 << ' ' << line2 << endl;

    return true;
}

Vec4f get3DLine(Vec4f line1, Vec4f line2) {
    Point2f p1(line1[0], line1[1]);
    Point2f p2(line1[2], line1[3]);

    Point2f p3(line2[0], line2[1]);
    Point2f p4(line2[2], line2[3]);

    if(getLineLength(line1) > 600)
        cout << "Matched0: " << line1 << ' ' << line2 << endl;


    double d1 = p1.x - p3.x;
    if(d1 <= 2) return Vec4f(0, 0, 0, 0);
    double z1 = fx * BASELINE / d1;
    double y1 = (imageSize.height / 2 - p1.y) * z1 / fx;
    double x1 = (p1.x - imageSize.width / 2) * z1 / fx;

    double d2 = p2.x - p4.x;
    if(d2 <= 2) return Vec4f(0, 0, 0, 0);
    double z2 = fx * BASELINE / d2;
    double y2 = (imageSize.height / 2 - p2.y) * z2 / fx;
    double x2 = (p2.x - imageSize.width / 2) * z2 / fx;

    if(getLineLength(line1) > 600)
        cout << "Matched1: " << line1 << ' ' << line2 << endl;

    if ((y1 > 1000 && y2 > 1000) || (y1 <= 5 && y2 <= 5)) return Vec4f(0, 0, 0, 0);
    
    if(getLineLength(line1) > 600)
        cout << "Matched2: " << line1 << ' ' << line2 << endl;

    if (abs(x2 - x1) < 2 && abs(z2 - z1) < 2) return Vec4f(0, 0, 0, 0);

    if (x1 > x2) {
        double temp = x1;
        x1 = x2;
        x2 = temp;

        temp = z1;
        z1 = z2;
        z2 = temp;
    }
       
    return Vec4f(x1, z1, x2, z2);
}

//라인 1에 대한 라인 2의 각도(반시계방향)
float getRadInLines(Vec4f line1, Vec4f line2) {
    Point2f v1(line1[2] - line1[0], line1[3] - line1[1]);
    Point2f v2(line2[2] - line2[0], line2[3] - line2[1]);

    float norm1 = sqrt(v1.x * v1.x + v1.y * v1.y);
    float norm2 = sqrt(v2.x * v2.x + v2.y * v2.y);
    if (norm1 == 0 || norm2 == 0) return 0.0f;

    v1.x /= norm1; v1.y /= norm1;
    v2.x /= norm2; v2.y /= norm2;

    float dot = v1.x * v2.x + v1.y * v2.y;
    float cross = v1.x * v2.y - v1.y * v2.x;

    return atan2(cross, dot);
}

Vec4f transformLine(Vec4f line, float radian, float scale) {
    float x0 = line[0], y0 = line[1];
    float x1 = line[2], y1 = line[3];

    float dx = x1 - x0;
    float dy = y1 - y0;

    float dx_rot = scale * (dx * cos(radian) - dy * sin(radian));
    float dy_rot = scale * (dx * sin(radian) + dy * cos(radian));

    float new_x1 = x0 + dx_rot;
    float new_y1 = y0 + dy_rot;

    return Vec4f(x0, y0, new_x1, new_y1);
}

float getLineError(Vec4f line1, Vec4f line2) {
    // 시작점 거리
    float dx1 = line1[0] - line2[0];
    float dz1 = line1[1] - line2[1];

    // 끝점 거리
    float dx2 = line1[2] - line2[2];
    float dz2 = line1[3] - line2[3];

    // 평균 거리 (또는 최대거리 써도 됨)
    float dist = sqrt(dx1 * dx1 + dz1 * dz1) + sqrt(dx2 * dx2 + dz2 * dz2);
    return dist / 2.0f;
}

bool isMergeCandidate(Vec4f line1, Vec4f line2) {
    Point2f c1((line1[0] + line1[2]) / 2.0, (line1[1] + line1[3]) / 2.0);
    Point2f c2((line2[0] + line2[2]) / 2.0, (line2[1] + line2[3]) / 2.0);


    float angleDiff = abs(getRadInLines(line1, line2));
    float centerDist = norm(c1 - c2);
    float lenA = getLineLength(line1);
    float lenB = getLineLength(line2);
    float lenRatio = lenA / lenB;
    return angleDiff < 0.1 && centerDist < ERROR_THRESHOLD && lenRatio > 0.9 && lenRatio < 1.1;
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

    int width = 3264, height = 2464, fps = 15;

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
    vector<pair<Vec4f, unsigned char> > linesOnMap;
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

            for (int i = 0; i < 3; i++) {
                Mat capture_left, capture_right;

                cap1.read(capture_left);
                cap0.read(capture_right);

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
            img_left /= 3.0;
            img_right /= 3.0;

            img_left.convertTo(img_left, CV_8UC1);
            img_right.convertTo(img_right, CV_8UC1);

            if (img_left.empty() || img_right.empty()) {
                cout << "Cannot read Image" << endl;
                return -1;
            }

            //flip(img_left, img_left, -1);
            //flip(img_right, img_right, -1);

            Ptr<CLAHE> clahe = createCLAHE(1.0, Size(16, 16));
            clahe->apply(img_left, img_left);
            clahe->apply(img_right, img_right);


            Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(LSD_REFINE_STD);
            vector<Vec4f> lines, lines_right;

            lsd->detect(img_left, lines);
            lsd->detect(img_right, lines_right);

            lines = mergeCollinearLines(lines);
            lines_right = mergeCollinearLines(lines_right);

            cout << "lines" << endl;
            for(auto l : lines){
                if(getLineLength(l) > 600){
                    cout << l << endl;;
                }
            }
            cout << "lines_right" << endl;
            for(auto l : lines_right){
                if(getLineLength(l) > 600){
                    cout << l << endl;;
                }
            }
            cout << endl;

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
            Ptr<ORB> orb = ORB::create(2000);
            orb->setFastThreshold(ORBTHRESHOLD);

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
            
            vector<bool> used_left, used_right;
            for (int i = 0; i < lines.size(); i++) {
                used_left.push_back(false);
            }
            for (int i = 0; i < lines_right.size(); i++) {
                used_right.push_back(false);
            }
            
            // 선분 매칭
            vector<Vec4f> current3DLines;
            for (int i = 0; i < lines.size(); i++) {
                if (used_left[i]) continue;

                vector<int> lineList_left, lineList_right;
                lineList_left.push_back(i);
                used_left[i] = true;

                for (int j = 0; j < lines.size(); j++) {
                    if (used_left[j]) continue;
                    if (isSameLine(lines[i], lines[j])) {
                        used_left[j] = true;
                        lineList_left.push_back(j);
                    }
                }

                for (int j = 0; j < lines_right.size(); j++) {
                    if (used_right[j]) continue;
                    if (isSameLine(lines[i], lines_right[j])) {
                        used_right[j] = true;
                        lineList_right.push_back(j);
                    }
                }

                for (auto j : lineList_left) {
                    float c1 = (lines[j][0] + lines[j][2]) / 2.0;
                    int best_idx = -1;
                    float best_dist = INT_MAX;
                    for (auto k : lineList_right) {
                        float c2 = (lines_right[k][0] + lines_right[k][2]) / 2.0;
                        if (c1 - c2 < 0 || c1 - c2 > LINEMATCHTHRESHOLD || c1 - c2 > best_dist) continue;
                        best_idx = k;
                        best_dist = c1 - c2;
                    }
                    if (best_idx == -1) continue;
                    Vec4f line = get3DLine(lines[j], lines_right[best_idx]);
                    if (line == Vec4f(0, 0, 0, 0)) continue;
                    
                    current3DLines.push_back(line);
                }
            }
            cout << "lines: " << current3DLines.size() << endl;
            

            vector<Vec4f> nearestLines;
            // 가장 가까운 선분 5개 추출 (회전 및 거리 계산에 사용)
            if (linesOnMap.size() != 0) {
                vector<pair<float, int> > z_idx;
                for (int i = 0; i < current3DLines.size(); i++) {
                    z_idx.push_back(make_pair((current3DLines[i][1] + current3DLines[i][3]), i));
                }

                sort(z_idx.begin(), z_idx.end());

                for (int i = 0; i < z_idx.size() && nearestLines.size() < 5; i++) {
                    if (z_idx[i].first <= 10) continue;

                    int idx = z_idx[i].second;
                    if (getLineLength(current3DLines[idx]) < 30) continue;
                    nearestLines.push_back(current3DLines[z_idx[i].second]);
                }
            }
            cout << "nearest: " << nearestLines.size() << endl;

            double rotate = 0.0;
            double dist = 1.0;
            double min_dx = self_x, min_dz = self_z;

            float minTotalError = 1e7;
            vector<pair<Vec4f, int>> newLines;
            if (nearestLines.size() != 0) {
                for (auto lom : linesOnMap) {
                    vector<pair<Vec4f, int>> currentLines;
                    double rad = getRadInLines(nearestLines[0], lom.first);
                    double ratio = getLineLength(lom.first) / getLineLength(nearestLines[0]);

                    double dx = self_x + lom.first[0] - nearestLines[0][0];
                    double dz = self_z + lom.first[1] - nearestLines[0][1];

                    float totalError = 0;
                    int matchCount = 1;
                    currentLines.push_back(make_pair(lom.first, 0));
                    for (int i = 1; i < nearestLines.size(); i++) {
                        Vec4f newLine = transformLine(nearestLines[i], rad, ratio);
                        newLine[0] += dx;
                        newLine[1] = dz - newLine[1]; 
                        newLine[2] += dx;
                        newLine[3] = dz - newLine[1];

                        float minErr = 1e8;
                        for (const auto& lom : linesOnMap) {
                            float err = getLineError(newLine, lom.first);
                            if (err < minErr)
                                minErr = err;
                        }

                        if (minErr < ERROR_THRESHOLD) {  // 예: 30cm 이내
                            totalError += minErr;
                            matchCount++;
                            currentLines.push_back(make_pair(newLine, i));
                        }
                    }

                    if (matchCount > nearestLines.size() * 0.66 && totalError < minTotalError) {
                        minTotalError = totalError;
                        rotate = rad;
                        dist = ratio;
                        min_dx = dx;
                        min_dz = dz;
                        newLines = currentLines;
                    }
                }
            }
            radian += rotate;
            radian = fmod(radian, 2 * 3.141592);
            if (radian < 0) {
                radian += 2 * 3.141592;
            }
            
           
            // 회전이 5도 이상일 시 orb가 아닌 선으로 위치 추정
            if (abs(rotate) >= 0.087) {
                int sum = 0;
                double nx = 0, nz = 0;
                for (auto line : newLines) {
                    double rcx = cos(radian) * nearestLines[line.second][0] - sin(radian) * nearestLines[line.second][1];
                    double rcz = sin(radian) * nearestLines[line.second][0] + cos(radian) * nearestLines[line.second][1];

                    // 카메라의 월드 좌표 추정
                    nx += line.first[0] - rcx;
                    nz += line.first[1] - rcz;
                    
                    sum++;
                    
                    rcx = cos(radian) * nearestLines[line.second][2] - sin(radian) * nearestLines[line.second][3];
                    rcz = sin(radian) * nearestLines[line.second][2] + cos(radian) * nearestLines[line.second][3];

                    // 카메라의 월드 좌표 추정
                    nx += line.first[2] - rcx;
                    nz += line.first[3] + rcz;

                    sum++;
                }
                nx /= sum;
                nz /= sum;

                self_x = static_cast<int>(round(nx));
                self_z = static_cast<int>(round(nz));
            }
               
            // orb 키포인트 매칭 (거리 계산에 사용)
            vector<pair<KeyPoint, int> > kp_matched;
            vector<DMatch> good_matches;
            for (const auto& m : matches) {
                if (m.distance < MATCHTHRESHOLD) {  // 임계값 적용
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

            // 회전이 5도 미만일 시 orb로 위치 추정
            if (abs(rotate) < 0.087) {
                vector<DMatch> matches_prev;
                if (desc_prev.rows != 0 && desc_matched.rows != 0)
                    matcher.match(desc_prev, desc_matched, matches_prev);

                for (const auto& m : matches_prev) {
                    if (m.distance < MATCHTHRESHOLD) {  // 임계값 적용
                        good_matches.push_back(m);
                    }
                }
                matches_prev = good_matches;


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
                        if (isOnMap(point_prev[m.queryIdx].second, point_prev[m.queryIdx].first))
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
            
            int prevSize = linesOnMap.size();
            // 선 병합 및 추가
            for (auto line : current3DLines) {
                Vec4f newLine = transformLine(line, rotate, dist);
                newLine[0] += min_dx;
                newLine[1] = min_dz - newLine[1];
                newLine[2] += min_dx;
                newLine[3] = min_dz - newLine[3];

                bool isMerged = false;
                for (int i = 0; i < prevSize; i++) {
                    if (isMergeCandidate(newLine, linesOnMap[i].first)) {
                        for (int j = 0; j < 4; j++) {
                            linesOnMap[i].first[j] += (newLine[j] - linesOnMap[i].first[j]) * BASIC_WEIGHT / linesOnMap[i].second;
                        }
                        if (linesOnMap[i].second + BASIC_WEIGHT < 256)
                            linesOnMap[i].second += BASIC_WEIGHT;

                        if (linesOnMap[i].first[0] > linesOnMap[i].first[2]) {
                            float temp = linesOnMap[i].first[0];
                            linesOnMap[i].first[0] = linesOnMap[i].first[2];
                            linesOnMap[i].first[2] = temp;

                            temp = linesOnMap[i].first[1];
                            linesOnMap[i].first[1] = linesOnMap[i].first[3];
                            linesOnMap[i].first[3] = temp;


                        }
                        isMerged = true;
                        break;
                    }
                }
                if (!isMerged) {
                    linesOnMap.push_back(make_pair(newLine, BASIC_WEIGHT));
                }
            }

            // 맵 구현
            for (auto line : linesOnMap) {
                Point p1(line.first[0], line.first[1]);
                Point p2(line.first[2], line.first[3]);

                LineIterator it(map, p1, p2);
                for (int i = 0; i < it.count; i++, ++it) {
                    Point pt = it.pos();
                    if (0 <= pt.y && pt.y < map.rows && 0 <= pt.x && pt.x < map.cols) {
                        map.at<uchar>(pt) = line.second;
                    }
                }
            }

            //맵 표시
            cvtColor(map, map_visual, COLOR_GRAY2BGR);
            if (0 <= self_x && self_x < map.cols && 0 <= self_z && self_z < map.rows)
                circle(map_visual, Point(self_x, self_z), 5, Scalar(0, 255, 0), -1);

            imshow("STEREO VISION V-SLAM Map", map_visual);


            cout << "pos: " << self_x << ' ' << self_z << endl;
            cout << "rotate: " << radian << endl;

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

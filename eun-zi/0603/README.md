## 0603 실험 결과 정리
노션 링크: https://www.notion.so/25-06-03-SLAM-2074fce89fa480b4abfcd798731edfd2?source=copy_link

1. 특징점 추출 기법을 orb에서 canny+Hough로 변경하여 매칭 성능을 비교하는 실험
2. 매칭에 시차 필터링 값을 적용하여 매칭 오류를 줄이는 실험

## 파일 구성

| 파일명                                 | 설명 |
|----------------------------------------|------------------------------------------------------------------------------|
| 'slam_pipeline_cannyHough_feature.cpp' | Canny + Hough 변환으로 검출한 선의 끝점을 특징점으로 사용하여 매칭을 수행하는 코드 |
| 'slam_pipeline_orb_feature.cpp'        | orb를 특징점으로 사용하여 매칭을 수행하는 코드                                   |
| 'slam_pipeline_match.cpp'              | 매칭에 시차 필터링을 적용하여 수행하는 코드                                      |
| 'feature_imgaes 폴더'     　     　   　| Canny + Hough 특징점과 orb 특징점 결과 이미지 　                         　     |
| 'matching_imgaes 폴더'     　     　  　| 매칭에 시차 필터링을 적용한 결과 이미지        　                         　     |


## 작성자

- 임은지 (eun-zi)


## 0605
매핑 관련해서 코드를 추가해 보고 있는데
나노를 움직이지 않아도 계속 스스로의 위치가 이동했다는 맵을 그리는 문제가 있음
카메라 노이즈로 같은 장면도 다른 특징점을 잡는 문제로
움직이지 않아도 이전 특징점과 현재 특징점을 같은 점으로 인식하지 못하기 때문에 그런 것으로 보임

일단 3초마다 사진을 찍고 맵을 갱신하도록 하였음
자신의 위치를 노란색 점으로 나타내고
보고 있는 곳을 파란 고깔로 나타내 어디에 있고, 어느 방향을 보고 있는지 설정해두었음
너무 멀리 있는 물체를 장애물로 인식할 필요가 있을까 라라는 의문이 들어 거리의 제한을 둘 것을 생각 중

젯슨 나노가 지나갈 수 있는 공간은 장애물로 표시할 필요가 없기 때문에 젯슨 나노의 (높이, 너비)를 반영하는 방법을 생각 중
카메라 화질 이슈가 제일 큰 것으로 보여 스마트폰으로 코드를 변경하여 진행해야할 것 같음!

그리고 3초마다 사진을 찍도록 했지만 이미지를 처리하고 맵을 그리는 곳에서 시간이 오래 걸리는지
사진 딜레이가 심하게 일어나서 이 부분도 수정이 필요할 것으로 보임!

📌 https://github.com/MrPicklesGG/ORB_SLAM3_Grid_Mapping  
📌 https://github.com/IATBOMSW/ORB-SLAM2_DENSE  
기존 orb slam들을 실행해 봐야할 거 같음

## 작성자

- 임은지 (eun-zi)

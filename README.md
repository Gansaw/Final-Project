# 고철장 작업차량 차량번호 인식 모델 및 웹서비스 개발

## 기술 스택 
- <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
- <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white"/> <img src="https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=React&logoColor=black"/>
- <img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white"/> <img width="23" src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Notion-logo.svg">
- <img src="https://img.shields.io/badge/Visual Studio Code-007ACC?style=flat-square&logo=Visual Studio Code&logoColor=white"/> <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat-square&logo=Google Colab&logoColor=white"/>


## 목차
1. [프로젝트 정의](#1-프로젝트-정의)
2. [팀원](#2-팀원)
3. [데이터 탐색](#3-데이터-탐색)
4. [모델 선정](#4-모델-선정)
5. [YOLOv5s](#5-YOLOv5s)
6. [ESRGAN](#6-ESRGAN)
7. [OCR](#7-OCR)
8. [Roboflow 3.0](#8-YOLOv5s)
9. [YOLOv5x](#9-YOLOv5s)  
10. [예측 결과](#10-예측-결과)

 
## 1. 프로젝트 정의
- 프로젝트명 : 고철장 작업차량 차량번호 인식 모델 및 웹서비스 개발
- 팀 구성 : 김재민(Back-End), 김은진(Front-End), 김민범(AI-Engineer), 최호진(AI-Engineer)
- 기술 분야 : Computer Vision, Deep Learning, Data Analysis & Processing
- 기간: 2023.08.21(월) ~ 2023.09.22(금) <br/>
  ![planning](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/07aeb058-d61f-4114-88b1-ca50e9159a99)

  
## 2. 팀원 
|<img width="200" alt="image" src="https://avatars.githubusercontent.com/u/129818813?v=4">|<img width="200" alt="image" src="https://avatars.githubusercontent.com/u/98063854?v=4">|<img width="200" alt="image" src="https://avatars.githubusercontent.com/u/70638717?v=4">|<img width="200" alt="image" src="https://avatars.githubusercontent.com/u/86204430?v=4">|
| :---------------------------------: | :-----------------------------------:| :---------------------------------: | :-----------------------------------:|
|                FrontEnd           |           Backend                       |              AI 모델 개발         |           AI 모델 개발                |       
|             김은진            |          김재민            |                          김민범                  |          최호진                      |      
|[GitHub](https://github.com/EUNJIN6131)|[GitHub](https://github.com/JaeMin1130)|[GitHub](https://github.com/sou05091/)|[GitHub](https://github.com/Gansaw/)|


## 3. 데이터 탐색
- 데이터 수집 : 현장 사진, 번호판 사진, 차량 입출입 로그 <br/>
![스크린샷 2023-09-25 100056](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/b95de212-ee46-407a-b0a0-2ba284efbd33)
![스크린샷 2023-09-18 182803](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/6d8e8f75-5cf5-407b-bbde-550c87201c31)
- 문제점 : 화질 문제, 라벨링 문제, 파일 형식 문제
- 해결방안 : 고해상도 모델 제작, 수작업 라벨링 진행, 확장자 추가 코드 제작
![스크린샷 2023-09-06 151045](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/1842e134-4b23-4ade-a27c-449c179c291d)
![스크린샷 2023-09-25 100344](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/fb3b2a1b-59ba-42e2-9466-06fed59b41e0)

 
## 4. 모델 선정
![스크린샷 2023-09-25 101336](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/2aa05642-5001-4ad6-b6af-3af657f91f94)
![스크린샷 2023-09-25 101357](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/d82c6808-d62f-4a6f-adea-09982f08808a)


## 5. YOLOv5s
- 데이터 라벨링 <br/>
![image](https://github.com/sou05091/MainProject_LicensePlate/blob/main/img/yolo/RoboFlow%20%EC%82%AC%EC%9A%A9.png)
- 데이터 전처리 & 증강 <br/>
![스크린샷 2023-09-25 100921](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/a670eb15-44a1-4552-b8e9-368a068631c1)
- 데이터셋 추출 <br/>
![image](https://github.com/sou05091/MainProject_LicensePlate/blob/main/img/yolo/Export.png)

#### Colab에서 모델 학습 진행 (라벨링 데이터 1000장 사용)
![image](https://github.com/sou05091/MainProject_LicensePlate/blob/main/img/yolo/model%20%ED%95%99%EC%8A%B5.png)

### 9. YOLOv5 모델 결과
#### 전체 이미지 객체 인식
![image](https://github.com/sou05091/MainProject_LicensePlate/blob/main/img/yolo/result.png)
#### 번호판 객체 이미지 저장
![image](https://github.com/sou05091/MainProject_LicensePlate/blob/main/img/yolo/result1.jpg)

### 10. 이미지 분류 데이터 전처리
#### 데이터 전처리
- 이미지 분류 진행 (7월달에 출입한 모든 차량, 64개 class사용)
- class간 데이터 불균형 발생 (이미지 수가 작은 class에 대해 데이터 증강 적용)
- 분류된 이미지 번호판 판단
- 이미지 화질 판단 수작업

![image](https://github.com/sou05091/MainProject_LicensePlate/blob/main/img/classfication/folder.png)

### 11. 이미지 모델 분류
#### 모델 제작 (모델 사용 안함)
- VGG16 사용
- 상세 코드는 GitHub 참조
#### 모델 결과
![image](https://github.com/sou05091/MainProject_LicensePlate/blob/main/img/classfication/result.png)

![image](https://github.com/sou05091/MainProject_LicensePlate/blob/main/img/classfication/result1.png)

##### 모델 사용 안하는 이유
- 새로운 차량이 있을시 새롭게 학습해야함
- 이미지 수가 적어 Overfiting의 경향이 보임

### 12. 최종 모델
- YOLOv5x FineTuning
- ESRGAN을 활용하여 이미지 화질 개선
- YOLOv5로 번호판의 번호예측
- EasyOCR, RoboflowOCR 기능을 추가적으로 구현
- 3가지 모델을 앙상블기법으로 결과 추론

### 13. Flask 제작
#### AI모델 정리
- 총 5가지 모델 사용 (YOLOv5, ESRGAN, YOLOv5, EasyOCR, RoboFlowOCR)
- 3가지 모델(YOLOv5, EasyOCR, RoboFlowOCR) return값 반환 (Json 형식)

#### 모델 작업 순서
- YOLOv5 번호판 객체인식 및 저장
- ESRGAN 이미지 선명도 조절 및 흑백사진으로 변환
- EasyOCR, RoboFlowOCR, YOLOv5로 번호 예측

![image](https://github.com/sou05091/MainProject_LicensePlate/blob/main/img/yolo/Flask%20%EC%82%AC%EC%9A%A9.png)
### 개발 일지 
<a href="https://shrub-snap-550.notion.site/CRUD-566be659b7bf4693a6515f408cf2f1d9?pvs=4">개발 일지 보러 가기  <img width="23" src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Notion-logo.svg"> </a>****

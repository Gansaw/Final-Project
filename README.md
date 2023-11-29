# 고철장 작업차량 차량번호 인식 모델 및 웹서비스 개발
D 기업에서는 고철장에 있는 트럭의 번호판을 식별하고 번호를 인식하여 DB에 저장되어 있는 차량의 정보를 조회하여 차량의 작업유무를 실시간으로 모니터링하는 시스템을 구축하고 있다. 하지만 CCTV로 촬영한 사진이라 화질이 좋지 않고 정면이 아닌 사선으로 촬영되어 번호를 인식하는데 많은 어려움을 가지고 있다고 한다. 이에 우리는 딥러닝을 이용하여 차량에 있는 번호판을 식별하고 이후 번호를 인식하여 차량 입출입을 관리하는 웹서비스를 개발하고자 한다.

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
8. [Roboflow_3.0](#8-Roboflow_3.0)
9. [YOLOv5x](#9-YOLOv5x)  
10. [Flask](#10-Flask)

 
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
![스크린샷 2023-09-25 103021](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/fd9e7cfa-77f7-4555-9eaf-ea621a01268f)
![image](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/36416825-ccae-43e0-aefe-888e5308ca04)
- 데이터 전처리 & 증강 <br/>
![스크린샷 2023-09-25 100921](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/a670eb15-44a1-4552-b8e9-368a068631c1)
- 데이터셋 추출 <br/>
![image](https://github.com/sou05091/MainProject_LicensePlate/blob/main/img/yolo/Export.png)
- 모델 학습 <br/>
![image](https://github.com/sou05091/MainProject_LicensePlate/blob/main/img/yolo/model%20%ED%95%99%EC%8A%B5.png)
- Object Detection <br/>
![image](https://github.com/sou05091/MainProject_LicensePlate/blob/main/img/yolo/result.png)
- 추출된 번호판 사진 저장 <br/>
![스크린샷 2023-09-25 101903](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/ca326dc0-4e08-41a2-bbef-98a841c2fd2f)


## 6. ESRGAN
- 저해상도 이미지를 고해상도로 변환
- grayscale 기능을 추가하여 모델의 학습 효율 증가
- 적용 결과 <br/>
![스크린샷 2023-09-25 102142](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/4594f619-c28f-4da8-b814-7b7a211de121)


## 7. OCR
- Easy-OCR 사용
- 예측 결과 1 <br/>
![스크린샷 2023-09-25 102513](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/d1dde896-9612-4775-8edb-e551debdb5a8)


## 8. Roboflow_3.0
- 데이터 라벨링 <br/>
![스크린샷 2023-09-25 103036](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/529e2e7e-87bf-4490-b05e-f95fc8c1247c)
![labeling2](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/1558bd34-788f-4d1c-aef0-524ad7c0ae7a)
- 데이터 전처리 & 증강 <br/>
![스크린샷 2023-09-25 102708](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/cfeec9b5-436f-4c4b-80cf-b97a82ca2f7c)
- 학습 결과 <br/>
F1-Score : 85.3% <br/>
mAP : 89.3%, precision : 86.3%, recall : 84.4% <br/>
![스크린샷 2023-09-05 174637](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/b8be6357-0935-4e44-97f3-738088132c42)
- 예측 결과 2 <br/>
![스크린샷 2023-09-25 103912](https://github.com/Gansaw/License_Plate_Recognition/assets/86204430/db044171-54d6-4d06-97fb-be26128ed30b)


## 9. YOLOv5x
- This model is made by MinBeom Kim
- More details, visit [김민범 Github](https://github.com/sou05091/MainProject_LicensePlate) <br/>


## 10. Flask
- Flask server is made by MinBeom Kim
- More details, visit [김민범 Github](https://github.com/sou05091/MainProject_LicensePlate) <br/>


## 개발 일지 
<a href="https://shrub-snap-550.notion.site/CRUD-566be659b7bf4693a6515f408cf2f1d9?pvs=4">개발 일지 보러 가기  <img width="23" src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Notion-logo.svg"> </a>****


## 라이센스(License)
This project is licensed by <a href = "https://creativecommons.org/licenses/by-nc-nd/4.0/">Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)</a> which is provided by Creative Commons. You do not allow commercial use, do not allow variations or derivatives, and require the original author's representation when sharing works publicly. <br/>

이 프로젝트는 Creative Commons에서 제공하는 <a href = "https://creativecommons.org/licenses/by-nc-nd/4.0/deed.ko">Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)</a> 라이센스가 적용되고 있습니다. 상업적 이용을 허용하지 않고, 변형 및 파생물 작성을 허용하지 않으며, 작품을 공개적으로 공유할 때 원저작자 표시를 요구합니다.

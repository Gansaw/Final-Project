# 차량 번호판 인식 AI 모델 개발, 차량 출입 관리 웹 서비스 개발하기

## 1. 프로젝트 소개 
- 개요: AI 모델을 통해 출입하는 차량의 번호판을 판별하고, 그 기록을 관리하는 웹서비스
- 활용 데이터 : 현장 CCTV 사진, 번호판 사진
- 기간: 2023.08.21(월) ~ 2023.09.22(금)

## 2. 팀원 
|<img width="200" alt="image" src="https://avatars.githubusercontent.com/u/129818813?v=4">|<img width="200" alt="image" src="https://avatars.githubusercontent.com/u/98063854?v=4">|<img width="200" alt="image" src="https://avatars.githubusercontent.com/u/70638717?v=4">|<img width="200" alt="image" src="https://avatars.githubusercontent.com/u/86204430?v=4">|
| :---------------------------------: | :-----------------------------------:| :---------------------------------: | :-----------------------------------:|
|                FrontEnd           |           Backend                       |              AI 모델 개발         |           AI 모델 개발                |       
|             김은진            |          김재민            |                          김민범                  |          최호진                      |      
|[GitHub](https://github.com/EUNJIN6131)|[GitHub](https://github.com/JaeMin1130)|[GitHub](https://github.com/sou05091/)|[GitHub](https://github.com/Gansaw/)|

## 3. 활용 스택 
<h3>Environment</h3>

- <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
- <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white"/> <img src="https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=React&logoColor=black"/>
- <img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white"/> <img width="23" src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Notion-logo.svg">
- <img src="https://img.shields.io/badge/Visual Studio Code-007ACC?style=flat-square&logo=Visual Studio Code&logoColor=white"/> <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat-square&logo=Google Colab&logoColor=white"/>

- Machine/Deap Learning:  TensorFlow, PyTorch

## 4. 주요 기능 
-  YOLOv5 활용 차량 번호판 객체 인식
-  출입 차량 번호판 이미지 추출
-  차량 번호판 이미지 분류
-  새로운 차량에 대해서 이미지 저장 후 학습 진행

## 5. 구현 모델
- ESRGAN을 활용한 번호판 이미지 전처리
- OCR을 활용한 번호판 번호 예측
- 전처리 후 CNN을 기반으로 한 Roboflow에서 모델 학습

## 6. 보고서 작성
- 프로젝트 진행 과정
- 모델 구현
- 프로젝트 결과 및 느낀 점
    
## 7. 개발 일지 
<a href="https://www.notion.so/02bdf271067b4de6bd30e72e18cc2522?pvs=4">개발 일지 보러 가기  <img width="23" src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Notion-logo.svg"> </a>****

## 8. 라이센스
This project is licensed by "Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)" which is provided by Creative Commons. Do not allow commercial use, do not allow variations or derivatives, and require the original author's representation when sharing works publicly.

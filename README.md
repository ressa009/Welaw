# Welaw
카카오 챗봇 기반의 생성형 AI 어시스턴스 
<br><br>


## 🖥️ 프로젝트 소개
다양한 비즈니스 모델에 연동할 수 있는 '챗봇 기반의 생성형 AI 어시스턴스'이며 법률회사 업무 도우미 역할을 수행하는 챗봇입니다.<br> GPT 3.5-turbo 모델이 적용되어 있어 풍부한 답변을 획득할 수 있고, OCR을 적용해 고객의 정보를 간편하게 저장할 수 있습니다. <br>또한 법률상담의 일환으로 새로운 판례, 사용자의 질문에 대해 대법원의 유사한 이전 판례 검색을 제공하며 해당 내용을 메일로 전송하는 서비스를 제공합니다. 
<br><br>

### 🕰️ 개발 기간
* 23.07.12 ~ 23.08.10

### 🧑‍🤝‍🧑 멤버구성
 - DB 3명, OCR 5명, PDF 3명

### ⚙️ 개발 환경
- `Python 3.10.9`
- **IDE** : Goorm GPU Server (NVIDIA Tesla T4)
- **Framework** : Quart(0.18.4)
- **Database** : MySQL

<br>

## 📌 주요 기능
#### 질문하기 - <a href="" >상세보기</a>
- Prompt 적용 : 
- 사용자 및 GPT 응답 DB 저장 - 맥락을 이해한 대화 가능 
#### 명함 OCR - <a href="" >상세보기</a>
- 이미지 전처리
- EASY OCR 라이브러리
- 리딩 내용 GPT로 분류 및 DB 저장, 수정 후 저장
#### 법륩상담 - <a href="" >상세보기</a>
- 유사 판례를 기반으로 법률 상담
- 유사 판례 검색 (1) 직접 판례 PDF를 업로드, (2) 서술을 기반으로 찾는 방식
- 판례 검색 결과 메일 전송
  
<br>

## 📑 시나리오
![image](https://github.com/ressa009/Welaw/assets/47082555/5e0ce4f0-c492-404e-bfc9-e88d88f479c0)

<br>

## 🎬 기능 화면
![메뉴](https://github.com/ressa009/Withmerry/assets/47082555/dea1fbe6-1768-43db-9fac-5b6b280636a5)
![질문하기](https://github.com/ressa009/Withmerry/assets/47082555/002a991e-c08a-4400-9360-4a64f0a25a90)<br>
![병원검색+방문후기](https://github.com/ressa009/Withmerry/assets/47082555/0d31c317-640d-4cd8-81d5-70ae18daf8da)
![진료예약](https://github.com/ressa009/Withmerry/assets/47082555/8fbaded4-ffdc-4d81-bc97-8dd23d6c652b)





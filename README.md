# recipe-recommendation-chatbot
<p align="center">
  <img src="https://github.com/kseob758/wtable-collection/assets/125840318/980639b5-f09b-41f5-b4c3-b160e69432fb" width="400" height=400>
</p>


## 프로젝트 개요
### 주제
- **Chat GPT API**를 이용하여 냉장고 안 재료로 만들 수 있는 **요리 레시피를 추천해주는 챗봇** 제작
  
### 주제 선정의 배경
- 경제적 측면 : 코로나19로 인한 보건위기가 사그라들었으나 외식비 상승과 배달료 인상으로 소비자들의 부담이 커지고 있음
- 편의적 측면 : 사람들은 귀찮고, 요리할 시간이 부족하고, 요리가 어렵다는 이유로 집밥을 해먹지 않음
  
### 기대효과
- **금전적 절약** 효과
- **남는 재료의 처리 문제** 해소
- **요리법을 일일이 찾아봐야 하는 번거로움** 해소
<br>
  
## 프로젝트 수행 절차 및 방법
<p align="center">
  <img src="https://github.com/kseob758/wtable-collection/assets/125840318/f194c16d-218e-4a74-8135-7973f460a406" width="900">
</p>

### 1. 데이터 수집
  - [우리의식탁](https://wtable.co.kr/recipes) 페이지에서 요리이름, 카테고리, 난이도, 소요시간, 상세페이지 링크, 이미지 링크, 재료, 요리방법, 요리설명 수집
  - requests, BeautifulSoup사용
  - 메인요리부터 오븐요리까지 18개 카테고리 활용

### 2. 전처리
- 요리 재료에 포함된 불용어 위주로 처리
  - 송송 썬 파 -> 파
  - 삶은 연근 -> 연근

### 3. 임베딩
- Hugging Face의 Sentence Transformer 이용
- 여러 임베딩 모델을 사용해본 후 [jhgan/ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask) 사용
- 임베딩 벡터로 변환한 파생변수 생성
  - 통깨 꼬시래기 식초 참기름 고춧가루... -> [0.32755455, 0.25685662, 0.5167652, 0.6005948,...

### 4. 모델링
- input 으로 사용자가 문자열을 입력하면 벡터화하여 기존의 임베딩 벡터와 코사인 유사도를 계산
- **코사인 유사도 상위 기준으로 output 출력(추천)**
- OpenAI API 를 활용하여 사용자의 문장 input의 **의도를 파악**하여 각각의 함수를 실행

### 5. streamlit 빌드
<br>

## 프로젝트 결과물
- 콜드 스타트 상황에서 상호작용을 통해 즉각적인 추천이 가능
- 사용자가 식재료를 입력하면, 기존의 레시피 데이터에서 해당 재료와 매칭된 요리 레시피를 추천
- 추천된 요리에 대한 상세 설명을 원할 경우 제공
- 추천된 요리에 대한 요리 방법을 원할 경우 제공
- 필요한 식재료 구매 추천 및 링크로 연결 가능
- 데모 영상 :  (link)

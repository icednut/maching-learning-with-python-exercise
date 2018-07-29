# Chapter 1 소개

- 머신러닝은 데이터에서 지식을 추출하는 작업
- 통계학, 인공지능, 컴퓨터 과학이 얽혀 있는 연구 분야이며 예측 분석이나 통계적 머신러닝으로도 불림
- 영화 추천, 음식 주문, 쇼핑, 맞춤형 온라인 라디오 방송, 사진에서 얼굴 찾기 등에 머신러닝 채택
- 이 책에서 소개하는 도구들은 별을 탐구하고 새로운 행성을 찾거나 새로운 미립자를 발견하고 DNA 서열을 분석, 맞춤형 암 치료법을 만드는 예제를  제공
- Chapter1에서는 머신러닝이 왜 유명해졌고 머신러닝을 사용해 어떤 문제를 해결할 수 있는지, 머신러닝 모델은 어떻게 만드는지 설명하고 머신러닝에서 사용하는 중요한 개념들을 소개

## 1.1 왜 머신러닝인가?

- 초창기 지능형 애플리케이션들은 하드코딩된 "if"와 "else" 명령을 사용하는 시스템
  - ex: 스팸 필터 (메일 블랙리스트를 만들어 처리)
  - 이는 규칙 기반 전문가 시스템rule-based expert system으로 볼 수 있음
  - 규칙은 사람이 수동으로 만듬 (모델링할 처리 과정을 사람이 잘 알고 있는 경우라고 볼 수 있음)
- 규칙을 직접 만들면 두 가지 단점이 생김
  - 결정에 필요한 로직은 한 분야, 작업에만 국한됨 (작업이 변경되면 전체 시스템을 다시 개발할 상황도 발생)
  - 규칙 설계에는 그 분야 전문가가 필요함

##### 머신러닝의 필요성

- 예를 들어 사람 얼굴 인식에는 일련의 규칙으로 표현하기가 근본적으로 불가능하다.
- 따라서 얼굴 이미지를 제공하면 얼굴을 특정하는 요소를 분석하는 방법이 필요

### 1.1.1 머신러닝으로 풀 수 있는 문제

- 지도 학습 supervised learning
  - 알고리즘에 입력과 기대되는 출력을 제공
  - 알고리즘은 주어진 입력에서 원하는 출력을 만드는 방법을 찾음
  - 학습된 알고리즘은 사람의 도움 없이도 새로운 입력이 주어지면 적절한 출력을 만듬
  - 요약하면 입력과 출력으로부터 학습하는 머신러닝 알고리즘들을 지도 학습 알고리즘
  - 지도 학습 예
    - 편지 봉투에 손으로 쓴 우편번호 숫자 판별
    - 의료 영상 이미지에 기반한 종양 판단
    - 의심되는 신용카드 거래 감지
- 비지도 학습 unsupervised learning
  - 여기서는 알고리즘에 입력은 주어지지만 출력은 제공되지 않음
  - 비지도 학습을 이해하거나 평가하기가 쉽지 않음
  - 비지도 학습의 예
    - 블로그 글의 주제 구분
    - 고객들을 취향이 비슷한 그룹으로 묶기
    - 비정상적인 웹사이트 접근 탐지
- 지도학습과 비지도 학습 모두 컴퓨터가 인식할 수 있는 형태로 입력 데이터를 준비해야함
  - 데이터를 엑셀 테이블이라고 생각하자
  - 우리가 판별해야할 개개의 데이터(개개의 이메일, 고객, 거래)는 행
  - 데이터를 구성하는 각 속성 (고객의 나이, 거래 가격, 지역)은 열
  - 고객이라면 나이, 성별, 계정 생성일, 온라인 쇼핑몰에서의 구매 빈도 등으로 표현
  - 흑백 이미지로 된 종양 데이터라면 크기, 모양, 색상의 진하기 등이 속성(feature)이 됨
- 머신러닝에서 하나의 개체 혹은 행을 **샘플**sample 또는 **데이터 포인트**data point라고 부름.
- 샘플의 속성, 즉 열을 **특성**feature 라고 함
- 나중에 이 책에서는 좋은 입력 데이터를 만들어내는 **특성 추출**feature extraction 혹은 **특성 공학**feature engineering라는 주제도 다룸

### 1.1.2 문제와 데이터 이해하기

- 사용할 데이터를 이해하고 그 데이터가 해결해야 할 문제와 어떤 관련이 있는지 이해하자
- 아무 알고리즘이나 선택해서 데이터를 입력해보는 것은 좋은 방법이 아님
- 뭘 할지를 먼저 이해해야한다 (머신러닝 알고리즘마다 잘 들어맞는 데이터나 문제의 종류가 다르다)
- 머신러닝을 적용하기 전 다음의 질문에 스스로 답을 해보자
  - 어떤 질문에 대한 답을 원하는가? 가지고 있는 데이터가 원하는 답을 줄 수 있는가?
  - 내 질문을 머신러닝의 문제로 가장 잘 기술하는 방법은 무엇인가?
  - 문제를 풀기에 충분한 데이터를 모았는가?
  - 내가 추출한 데이터의 특성은 무엇이며 좋은 예측을 만들어낼 수 있을 것인가?
  - 머신러닝 애플리케이션의 성과를 어떻게 측정할 수 있는가?
  - 머신러닝 솔루션이 다른 연구나 제품과 어떻게 협력할 수 있는가?

## 1.2 왜 파이썬인가?

- 파이썬에 데이터 적재, 시각화, 통계, 자연어 처리, 이미지 처리 등에 필요한 라이브러리가 있으며 사용하기 편함
- 일례로 터미널이나 주피터 노트북과 같은 도구로 대화하듯 프로그래밍할 수 있음

## 1.3 scikit-learn

- scikit-learn은 매우 인기 높고 독보적인 파이썬 머신러닝 라이브러리
- 잘 알려진 머신러닝 알고리즘들은 물론 알고리즘 설명 문서와 예제가 풍부함

### 1.3.1 scikit-learn 설치

- scikit-learn은 NumPy와 SciPy를 사용
- 그래프를 그리기 위해 matplotlib
- 대화식 개발을 위해 IPython과 Jupyter Notebook 설치 필요
- 필요한 패키지들을 모아놓은 파이썬 배포판을 설치하는 방법을 권장
  - Anaconda
  - Enthought Canopy
  - Python (x, y)
- 파이썬을 이미 설치했다면 아래 명령어로 필수 도구 설치 진행

```shell
pip install numpy scipy matplotlib ipython scikit-learn pandas pillow
```

## 1.4 필수 라이브러리와 도구들

- scikit-learn은 파이썬 과학 라이브러리인 NumPy와 SciPy를 기반으로 만듦
- pandas와 matplotlib, Jupyter Notebook도 사용할 예정

### 1.4.1 주피터 노트북

- 파이썬 코드를 브라우저에서 실행해주는 대화식 개발 환경
- 이 책에 포함된 모든 예제는 주피터 노트북으로 개발

### 1.4.2 NumPy

- 다차원 배열을 위한 기능과 선형 대수 연산, 푸리에 변환과 같은 고수준 수학 함수와 유사 난수 생성기를 포함
- scikit-learn에서 NumPy 배열은 기본 데이터 구조
- 우리가 사용할 데이터는 모두 NumPy 배열로 변환 필요
- NumPy의 핵심 기능은 다차원(n-차원) 배열인 ndarray 클래스 - 이 배열의 모든 원소는 동일한 데이터 타입이어야 함

```python
import numpy as np

x = np.array([[1,2,3], [4,5,6]])
print("x:\n{}".format(x))
```

### 1.4.3 SciPy

- SciPy는 과학 계산용 함수를 모아놓은 파이썬 패키지
- 고성능 선형 대수, 함수 최적화, 신호 처리, 특수한 수학 함수와 통계 분포 등을 포함한 많은 기능 제공
- 일례로 scipy.sparse가 있음
  - 이 모듈은 희소 행렬 기능을 제공

```python
from scipy import sparse

eye = np.eye(4)
print("NumPy 배열:\n{}".format(eye))
```

```python
sparse_matrix = sparse.csr_matrix(eye)
print("SciPy의 CSR 행렬:\n{}".format(sparse_matrix))
```

- Compressed Sparse Row는 0이 모두 채워진 2차원 배열을 만들지 않아 메모리가 부족할 수 있음
- 따라서 희소 행렬을 직접 만들 수 있어야 함 -> COO 포맷(Coordinate 포맷)을 이용해 희소행렬 만들수 있어야 함

```python
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO 표현:\n{}".format(eye_coo))
```

### 1.4.4 matplotlib

- 파이썬의  대표적인 과학 계산용 그래프 라이브러리
- 선 그래프, 히스토그램, 산점도 등을 지원

```python
%matplotlib inline
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker="x")
```

### 1.4.5 pandas

- 데이터 처리와 분석을 위한 파이썬 라이브러리
- R의 data.frame을 본떠서 설계한 DataFrame이라는 데이터 구조 기반
  - pandas의 DataFrame은 엑셀의 스프레드시트와 비슷한 테이블 형태 자료 구조
  - SQL처럼 테이블에 쿼리나 조인을 수행할 수 있음
  - pandas는 각 열의 타입이 달라도 됨
  - SQL, 엑셀 파일, CSV 파일 같은 다양한 파일과 데이터베이스에서 데이터를 읽을 수 있음

```python
from IPython.display import display
import pandas as pd

data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location': ["New York", "Paris", "Berlin", "London"],
        'Age': [24, 13, 53, 33]
       }
data_pandas = pd.DataFrame(data)
display(data_pandas)
```

```python
display(data_pandas[data_pandas.Age > 30])
```

### 1.4.6 mglearn

- 그래프나 데이터 적재와 관련한 세세한 코드를 일일이 쓰지 않아도 되게끔 이 책을 위해 만든 유틸리티 함수들
- 이 책에서는 간단하게 그림을 그리거나 필요한 데이터를 바로 불러들이기 위해 사용

> ##### 이 책의 모든 코드는 다음의 라이브러리를 기본적으로 임포트 한다고 가정
>
> ```python
> from IPython.display import display
> import numpy as np
> import matplotlib.pyplot as plt
> import pandas as pd
> import mglearn
> ```

## 1.5 파이썬 2 vs. 파이썬 3

파이썬 3 짱짱맨!

## 1.6 이 책에서 사용하는 소프트웨어 버전

```python
import sys
print("파이썬 버전: {}".format(sys.version))

import pandas as pd
print("pandas 버전: {}".format(pd.__version__))

import matplotlib
print("matplotlib 버전: {}".format(matplotlib.__version__))

import numpy as np
print("NumPy 버전: {}".format(np.__version__))

import scipy as sp
print("SciPy 버전: {}".format(sp.__version__))

import IPython
print("IPython 버전: {}".format(IPython.__version__))

import sklearn
print("scikit-learn 버전: {}".format(sklearn.__version__))
```

- scikit-learn은 가능한 최신 버전 유지하기

## 1.7 첫 번째 애플리케이션: 붓꽃의 품종 분류

- **문제: 한 아마추어 식물학자가 사용할 붓꽃의 품종을 알려주는 프로그램 작성하기**
- 보유한 데이터
  - 붓꽃의 꽃잎(petal)과 꽃받침(sepal)의 폭과 길이를 센티미터 단위로 측정한 데이터
  - setosa, versicolor, virginica 종으로 분류한 붓꽃의 측정 데이터
  - 이 측정 값을 이용해 채집한 붓꽃이 어떤 품종인지 구분
  - 앞으로 채집할 붓꽃은 이 세 종류뿐이라고 가정
- 붓꽃의 품종을 정확하게 분류한 데이터를 가지고 있으므로 이 문제는 **지도학습**
- 이 문제는 분류Classification 문제임
- 출력될 수 있는 값(붓꽃의 종류)들은 **클래스**class
- 이 문제는 세 개의 클래스를 분류하는 문제
- 특정 데이터 포인트에 대한 출력, 즉 품종은 **레이블**label

### 1.7.1 데이터 적재

```python
from IPython.display import display
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

iris_dataset = load_iris()
```

```python
print("iris_dataset의 키: \n{}".format(iris_dataset.keys()))
```

```python
print(iris_dataset['DESCR'][:193] + "\n...")
```

```python
print("타깃의 이름: {}".format(iris_dataset['target_names']))
```

```python
print("특성의 이름: \n{}".format(iris_dataset['feature_names']))
```

```python
print("data의 타입: {}".format(type(iris_dataset['data'])))
```

```python
print("data의 크기: {}".format(iris_dataset['data'].shape))
```

```python
print("data의 처음 다섯 행:\n{}".format(iris_dataset['data'[:5]]))
```

```python
print("target의 타입: {}".format(type(iris_dataset['target'])))
```

```python
print("target의 크기: {}".format(iris_dataset['target'].shape))
```

```python
print("타깃:\n{}".format(iris_dataset['target']))
```

### 1.7.2 성과 측정: 훈련 데이터와 테스트 데이터

- 앞의 데이터로 머신러닝 모델을 훈련하고 새로운 데이터의 품종을 예측
- 모델에 새 데이터를 적용하기 전에 잘 작동하는지 평가할 필요가 있음
- 모델은 훈련에 사용한 데이터를 기억하고 있는데 평가에 훈련 데이터를 사용하면 안됨 (이렇게 데이터를 기억한다는 것은 모델을 **일반화**하지 않았다는 뜻. 새로운 데이터에 대해서는 잘 작동하지 않는다는 말고 같음)
- 150개의 붓꽃 데이터를 두 그룹으로 나눔
  - **훈련 데이터**(훈련 세트training set)
  - **테스트 데이터**(테스트 세트, 홀드아웃 세트)
  - scikit-learn에서 데이터셋을 섞어서 나눠주는 train_test_split 함수를 사용
    - 이 함수는 전체 행 중 75%를 레이블 데이터와 함께 훈련 세트로 선정
    - 나머지 25%는 레이블 데이터와 함께 테스트 세트로 선정
    - 보통 테스트 데이터는 25%를 선정하는게 일반적임
- scikit-learn에서 데이터는 대분자 X, 레이블은 소문자 y로 표기

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'],
    iris_dataset['target'],
    random_state = 0
)
```

```python
print("X_train의 크기: {}".format(X_train.shape))
print("y_train의 크기: {}".format(y_train.shape))
```

```python
print("X_test의 크기: {}".format(X_test.shape))
print("y_test의 크기: {}".format(y_test.shape))
```

### 1.7.3 가장 먼저 할 일: 데이터 살펴보기

- 머신러닝 모델을 만들기 전 할 일
  - 머신러닝이 없이도 풀 수 있는 문제는 아닌지,
  - 흑은 필요한 정보가 누락되지는 않았는지
  - 비정상적인 값이나 특이한 값이 있는지 (예: 붓꽃 데이터 중 일부는 센티미터가 아니고 인치로 되어 있는 경우)
  - 를 알기 위해 데이터를 조사해봐야함
- 이런 것을 알기 위해 데이터를 시각화할 필요가 있음 -> 여기서는 산점도scatter plot가 좋음
  - 컴퓨터 화면은 2차원이기 때문에 2개의 특성(feature)만 사용
  - 나중에 3차원 산점도도 나옴

```python
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(
    iris_dataframe, 
    c=y_train, 
    figsize=(15,15), 
    marker='o', 
    hist_kwds={'bins': 20},
	s=60,
    alpha=.8,
    cmap=mglearn.cm3
)
```

### 1.7.4 첫 번째 머신러닝 모델: k-최근접 이웃 알고리즘

- 이제 실제 머신러닝 모델을 트레이닝
- 여기서는 분류 머신러닝 알고리즘으로 **k-최근접 이웃**k-Nearest Neighbors, k-NN 분류기를 사용
- k-NN 알고리즘
  - k는 가장 가까운 이웃 '하나'가 아니라 훈련 데이터에서 새로운 데이터 포인트에 가장 가까운 'k개'의 이웃을 찾는다는 뜻
  - 예를 들면 가장 가까운 세 개 혹은 다섯 개의 이웃
  - 그런 다음 이 이웃들의 클래스 중 빈도가 가장 높은 클래스를 예측값으로 사용
  - 지금은 하나의 이웃만 사용
- scikit-learn의 모든 머신러닝 모델은 Estimator라는 파이썬 클래스로 각각 구현

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
```

- knn 객체
  - 훈련 데이터로 모델을 만들고 새로운 데이터 포인트에 대해 예측하는 알고리즘을 캡슐화한 것
  - KNeighborsClassifier에 훈련 데이터 자체를 저장하고 있음
- 훈련 데이터셋으로부터 모델 만들기
  - 아래 코드를 실행하면 knn 객체 자체를 반환하고 실행 주체 knn 객체도 변경시킴

```python
knn.fit(X_train, y_train)
```

### 1.7.5 예측하기

- 레이블을 모르는 새 데이터에 대해 예측하기

```python
X_new = np.array([[5,2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))
```

```python
prediction = knn.predict(X_new)
print("예측: {}".format(prediction))
print("예측한 타깃의 이름: {}".format(iris_dataset['target_names'][prediction]))
```

- 이렇게 예측한 결과를 신뢰할 수 있을까?

### 1.7.6 모델 평가하기

- 테스트 데이터에 있는 붓꽃의 품종을 예측하고 실제 레이블(품종)과 비교하자

- 얼마나 많은 붓꽃 품종이 정확히 맞았는지 **정확도**를 계산하여 모델의 성능을 평가

- ```python
  y_pred = knn.predict(X_test)
  print("테스트 세트에 대한 예측값:\n {}".format(y_pred))
  ```

  ```python
  print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))
  ```

## 1.8 요약 및 정리

- 지도 학습과 비지도 학습 차이 설명
- 이 책에서 사용할 도구 간략하게 둘러보기
- 붓꽃의 품종 예측 실습
**`pandas` 기초 함수 정리**

___

`pandas`란?
- 관계형, 레이블 데이터를 다루기 위한 라이브러리

### Series
> 1차원 배열을 다루기 위한 자료구조
```python
import pandas as pd

pd.Series(numpy array / list, dtype = , index =)
```

- `numpy`의 배열이나 파이썬 `list`를 parameter로 받아 `dtype`으로 데이터 타입을 지정하거나 `index`로 indexing하여 Series를 생성할 수 있다.
- **하나의 데이터 타입만 저장하는 것이 아니라, 여러 데이터 타입의 데이터를 섞어 저장할 수 있다.**
  - 이 경우, data type은 `object`가 된다.
  - `index`를 지정하지 않으면, 0부터 indexing된다.

`Series` 내에, `indexing`을 위해 boolean, array 등을 이용할 수 있다.

#### `attributes`

- `shape`: 1차원이기 때문에 **(d, ) 형식**으로 출력된다.

- `np.nan`: 데이터의 값이 정해지지 않았을 때 사용한다.
- `values`: Series에서 데이터의 값만 `numpy array` 형태로 가져온다.

#### `useful function`

- `isnull(), isna()`: Series내의 nan 값의 위치를 True
- `notnull(), notna()`: Series내의 nan 값이 아닌 데이터의 위치를 True

`Example`
``` python
s.isna()

[출력]
0    False
1    False
2     True
3    False
4    False
dtype: bool
```

___ 
### DataFrame

> 2차원 배열을 다루기 위한 자료구조
```python
import pandas as pd

pd.DataFrame(numpy array / list, columns = )
```

- DataFrame에 parameter로 사용되는 `array`나 `list`는 2차원 형태여야 한다.
-  `columns = 1차원 배열`을 통해 column명을 지정할 수 있다.
-  추가로, `dictionary`를 이용할 수도 있다.
   -  이 경우, `key`가 `column`이 된다.

`Example`
```python
pd.DataFrame([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]], columns=['가', '나', '다'])

[출력]

    가	나	다
0	1	2	3
1	4	5	6
2	7	8	9

data = {
    'name': ['Kim', 'Lee', 'Park'], 
    'age': [24, 27, 34], 
    'children': [2, 1, 3]
}
pd.DataFrame(data)

[출력]

    name	age	    children
0	Kim	    24	    2
1	Lee	    27	    1
2	Park	34	    3
```

#### `attributes`

- `.T`: DataFrame을 Transpose 할 수도 있다.

- `index`: 기본 값으로 RangeIndex
  - `DataFrame.index = 1차원 배열` 형태로 row명을 지정할 수 있다.

- `columns`: column 명
  - 특히, DataFrame[column 명] 을 이용하여 indexing 할 수 있다.
  - 특정 col에 해당되는 row만을 가져올 수 있다.

- `values`: numpy array형식의 데이터 값

- `dtypes`: column 별 데이터 타입

#### `useful function`

- `rename(columns={'바꾸고자 하는 컬럼명': '바꿀 컬럼명'}, inplace=True)`: column 명을 변경
  - `inplace` 옵션은 변경 사항을 바로 적용하는데 사용된다.

___

## 파일 입출력

### 1. Excel

```python
excel = pd.read_excel('data/seoul_transportation.xlsx', sheet_name='버스', engine='openpyxl')

excel.head()
```
- `read_excel()` 함수를 이용하여, `sheet_name`에 맞는 sheet를 불러올 수 있다.

- `engine = openpyxl` 옵션은 read_excel이 실패할 시 추가해주면 된다.

- `sheet_name` = None 이라면 모든 sheet의 데이터를 가져온다.
- `head()` 함수는 해당 엑셀 파일의 데이터를 출력한다.


```python
excel.keys()

excel.to_excel('sample1.xlsx', index=False, sheet_name='샘플')
```


- `keys()`는 해당 엑셀 파일이 포함하고 있는 **시트를 조회**한다.

- `to_excel()` 함수는 DataFrame을 **excel 파일로 저장**한다.
- `index =` 옵션을 **false**로 설정하지 않으면 각 index(row) 가 별도의 column으로 저장되기 때문에 웬만하면 False로 설정하는 편이다.
- `sheet_name`을 통해 **sheet name**을 지정할 수 있다.

### 2. CSV

```python
df = pd.read_csv('data/seoul_population.csv')

df.to_csv('sample.csv', index=False)
```

- `read_csv()` 함수를 통해 csv 파일을 다룰 수 있다.

- `to_csv()` 함수를 통해 excel 파일이나, dataframe을 csv 파일 형식에 맞게 저장할 수 있다.

___ 
## 데이터 분석

> DataFrame에서 데이터를 분석하는데 사용할 수 있는 함수를 알아보자.

**`useful function`**

- `head()`: Default로 **상위 5개 행**을 보여주거나, parameter로 지정된 개수만큼의 행을 보여준다.

- `tail()`: Default로 **하위 5개 행**을 보여주거나, parameter로 지정된 개수만큼의 행을 보여준다.

- `info()`: 각 column의 Nan이 아닌 **데이터의 개수, 데이터 타입을 알려주는 표** 반환

- `value_counts()`: **해당 column에 포함된 데이터의 종류와 개수**를 표 형태로 반환

- `sort_index(ascending=True)`: 기본적으로 index 기준 오름차순으로 정렬
  - `ascending=False` 이라면, 내림차순으로 정렬

- `astype('dtype')`:dtype에 맞게 데이터 타입을 변경 

- `sort_values(by=['fare', 'age'], ascending=[False, True])`
  - `by =` 에 속한 column의 데이터 값에서 `ascending=`이 False라면 내림차순, True라면 오름차순 기준 정렬

- `loc(row slicing, column slicing)`: `index slicing` 과 `column slicing` 에 맞는 데이터 반환
  ```python
  cond1 = (df['fare'] > 30)

  # 조건2 정의
  cond2 = (df['who'] == 'woman')

  df.loc[cond1 & cond2]
  ```
  - 위 경우처럼, **boolean을 이용**할 수도 있다.

- `iloc()`: `loc()` 과 동일한 기능을 수행하지만, **index만을 허용한다.**

- `isin(data)`: data에 해당되는 값이 들어있는 행만을 반환한다.

## 통계

- `describe()`: Numerical column에 대한 정보를 표 형태로 제공해준다.
  - **min, max, count, mean, std**에 대한 정보가 담겨있다.
  - `include = object` 옵션을 사용하면 문자열 column에 대한 정보도 확인할 수 있다.

- `count()`: DataFrame에 들어있는 **각 column의 data 개수**를 반환한다.

- `mean()`: Column에 대한 **평균**을 계산한다.
  -  `skipna=False` 옵션을 사용하면, Nan이 있는 column은 Nan으로 표시된다.
- `median()`: Column에 대한 **중앙**값을 계산한다.
- `sum()`: Column에 대한 **총합**을 계산한다.
- `cumsum()`: Column에 대한 **누적합**을 계산한다.
- `cumprod()`: Column에 대한 **누적곱**을 계산한다.
- `var()`: Column에 대한 **분산**을 계산한다.
- `std()`: Column에 대한 **표준편차**를 계산한다.
- `max()`: Column에 대한 **최댓값**을 계산한다.
- `min()`: Column에 대한 **최솟값**을 계산한다.
- `agg(column, 함수)`: 여러 개의 통계 함수를 동시에 사용할 수 있도록 한다.
    ``` python
    df[['age', 'fare']].agg(['min', 'max', 'count', 'mean'])
    ```
    - `age`, `fare` column에 **min, max, count, mean** 함수를 적용한 결과를 반환한다.
- `quantile()`: 분위수 확인
  - `Quantile`이란? 주어진 데이터를 **동등한 크기로 분할**하는 지점
  - n개의 데이터가 있을 때, `quantile(0.8)`하게 되면 상위 80% 지점의 데이터를 반환한다.
- `unique()`: Column에서 중복되지 않은 데이터의 값
- `nunique()`: Column에서 중복되지 않은 데이터의 개수
- `mode()`: Column에 대한 최빈값을 계산한다.
- `corr()` - Column별 상관관계
    - **-1~1 사이의 범위**를 가집니다.
    - -1에 가까울 수록 두 column은 `반비례 관계`, 1에 가까울수록 `정비례 관계`를 의미합니다.
    - `corr()`은 **column 개수 x column 개수** 형태의 표를 반환하기 때문에, 특정 column과 나머지 column에 대한 정보만을 확인하고 싶다면 `df.corr()['survived']` 같이 작성해야 한다.

## 추가 유용한 함수
- `copy()`: 기존 DataFrame을 복사한 복제본을 생성. 복제본에 대한 **수정은 원본에 영향을 미치지 않는다.**

- `fillna()`: Nan에 해당하는 값을 특정 값으로 채울 수 있다.
    - argument에 사용된 값으로 채워진다.
    - `통계 함수`를 사용해서 채울 수도 있다.
    - 이때, 최빈값의 경우에는 `fillna(df[].mode()[0])`와 같이 0번째 index를 사용해야 한다.
- `dropna()`: Nan이 포함된 행을 제거하고 반환한다.

## 데이터 전처리

DataFrame에 없는 column명으로 `df['column name'] = value`를 사용하면 새로운 column을 생성할 수 있다.

- `insert(컬럼인덱스, 컬럼명, 값)`: 새로운 column을 생성한다.

- `drop(index)`: index에 따라 행을 삭제한다. index는 `DataFrame`에 사용 가능한 indexing 방법 모두 사용 가능하다.

- `drop(column name, axis = 1)`: `axis = 1` 옵션 사용 시, column 제거가 가능하다.

**Column간 연산**
> 단순히 column 별로 `+` 연산자를 통해, 값을 더하거나 문자열을 이어 붙일 수 있다.
- 이때, 여러 column 중 하나의 column이라도 Nan 값을 가지고 있다면 연산 결과는 Nan으로 처리된다.

- 연산 시, `round(숫자, 소수 몇 째자리)` 를 통해 소수점 아래 자릿수도 지정할 수 있다.


#### `useful functions`

- `pd.cut(x, bins, right=True)`: x를 bins 내에 있는 구간 별로 나눈다.

- `qcut()`: `cut()`와 동일한 기능, 각 그룹 내의 데이터 개수가 동일하도록 한다. 

- `concat()` 2개 이상의 DataFrame을 행 혹은 열 방향으로 연결

- `merge()` 2개의 DataFrame을 특정 Key를 기준으로 병합


## Group By
> DataFrame에서 특정 column이 기준 row가 된 표로 나타날 수 있도록 한다.

`Example`
```python
df.groupby(['sex', 'pclass']).mean()


[출력]
            survived	age	        sibsp	    parch	
sex  pclass 							
female	1	0.968085	34.611765	0.553191	0.457447	
        2	0.921053	28.722973	0.486842	0.605263	
        3	0.500000	21.750000	0.895833	0.798611	
male	1	0.368852	41.281386	0.311475	0.278689	
        2	0.157407	30.740707	0.342593	0.222222	
        3	0.135447	26.507589	0.498559	0.224784
```
## Pivot Table

> GroupBy + Aggregation 을 수행하는 함수
```
df = pd.DataFrame({
    "과목": ["수학", "수학", "영어", "영어"],
    "반":   ["A", "B", "A", "B"],
    "점수": [80, 90, 70, 85]
})

pd.pivot_table(df, values="점수", index="과목", columns="반", aggfunc="mean")

[결과]
반     A     B
과목
수학  80.0  90.0
영어  70.0  85.0
```

## 데이터 시각화

> `plot()` 함수 내에 `kind=` 옵선을 다르게 설정하여 사용할 수 있다.

사용 가능한 옵션은 아래와 같다.

- `line`: 선그래프

- `bar`: 바 그래프

- `barh`: 수평 바 그래프

- `hist`: 히스토그램

- `kde`: 커널 밀도 그래프

- `hexbin`: 고밀도 산점도 그래프

- `box`: 박스 플롯

- `area`: 면적 그래프

- `pie`: 파이 그래프

- `scatter`: 산점도 그래프



___

`Reference`: https://wikidocs.net/book/4639
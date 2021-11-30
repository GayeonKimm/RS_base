import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.metrics.pairwise import cosine_similarity

print(os.listdir("C:/Users/user/Desktop/kaggle/the-movies-dataset/"))

path = "C:/Users/user/Desktop/kaggle/the-movies-dataset/"

data = pd.read_csv(path+'movies_metadata.csv', low_memory=False)
print(data.head(2))

# 변수들 파악
print(data.columns)

# 전처리
# overview의 결측치가 있는 항목은 모두 제거
data = data[data['overview'].notnull()].reset_index(drop=True)
print("\n결측치 제거\n", data.shape)

# 데이터 슬라이싱. 필요한 것만 가지고 가려고
data = data.loc[:20000].reset_index(drop=True)

# 불용어 제거. 유의미하지 않은 단어 토큰 제거
tfidf = TfidfVectorizer(stop_words='english')
# 여기까진 객체 생성 과정임

# overview에 대해서 tf-idf 수행
tfidf_matrix = tfidf.fit_transform(data['overview'])
print("\noverview에 대해 수행한 결과 \n", tfidf_matrix.shape)

cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("\n유사도 계산 결과\n",cosine_matrix.shape)

np.round(cosine_matrix, 4)
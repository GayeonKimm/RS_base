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
print("\n overview에 대해 수행한 결과 \n", tfidf_matrix.shape)

cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("\n 유사도 계산 결과\n",cosine_matrix.shape)

np.round(cosine_matrix, 4)



# movie title 과 id를 매핑할 dictionary 생성 (movie to id )
movie2id = {}
for i,c in enumerate(data['title']) :
    movie2id[i] = c

# id와 movie title을 매필할 dictionary 생성
id2movie = {}
for i,c in movie2id.items():
    id2movie[c] = i


# toy Story의 id 추출
idx = id2movie['Toy Story']  # Toy story가 0번 인덱스임
sim_scores = [(i,c) for i, c in enumerate(cosine_matrix[idx]) if i !=idx]
# 자기 자신을 제외한 영화들의 유사도 및 인덱스를 추출

sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse=True)
# 유사도가 높은 순서대로 정렬 # lambda

print("\n 상위 10개의 인덱스와 유사도 추출\n", sim_scores[0:10])


# 열개만 뽑기
sim_scores = [(movie2id[i], score) for i ,score in sim_scores[0:10]]
print(sim_scores)



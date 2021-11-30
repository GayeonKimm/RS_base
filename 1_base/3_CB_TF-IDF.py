# 0. data 생성

docs =  [
    '먹고 싶은 사과',       # 문서 0
    '먹고 싶은 바나나',     # 문서 1
    '길고 노란 바나나 바나나',  # 문서 2
    '저는 과일이 좋아요'     # 문서 3
]
print("\n 0. data 생성 \n", docs)


# 1. Counter Vectorizer 객체 생성
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

# 2. 문장을 Counter Vectorizer 형태로 변형
countvect = vect.fit_transform(docs)
print('\n 문장을 counter vectorizer 형태로 변형\n', countvect)

# 3. toarray() 를 통해서 문장이 Vector 형태릐 값을 얻을 수 있음
countvect.toarray()
print("\n 문장이 vector 형태로 얻을 수 있게\n", countvect.toarray())

# 하지만 각 인덱스와 컬럼이 무엇을 의미하는지 알 수 없음
print('\n dic 형태로 출력\n', vect.vocabulary_)

print('\n 이를 정렬하여 출력\n', sorted(vect.vocabulary_))


# 4. dataframe 형태로 정리하여 출력
import pandas as pd
countvect_df = pd.DataFrame(countvect.toarray(), columns=sorted(vect.vocabulary_))
countvect_df.index = ['문서1', '문서2', '문서3', '문서4']
print('\n dataframe 형태로 출력\n', countvect_df)


# 5-1. 유사도 계산
from sklearn.metrics.pairwise import cosine_similarity
count_df_si = cosine_similarity(countvect_df, countvect_df)
print("\n cosine similarity 계산\n", count_df_si)

# 0번 문서는 1번과 유사하다는 결론을 얻을 수 있음.
# 동일한 방식으로 TF-IDF 를 수행하면 아래와 같음

# 5-2. TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
tfvect = vect.fit(docs)
print('\n TF-IDF 객체 생성 후 적용\n', tfvect )


# DataFrame 형태로 변환
tfidv_df = pd.DataFrame(tfvect.transform(docs).toarray(), columns=sorted(vect.vocabulary_))
tfidv_df.index = ['문서1', '문서2','문서3', '문서4']
print('\n TF-IDF 객체 생성 후 적용\n', tfidv_df)

# 유사도 계산
tfidv_df_si = cosine_similarity(tfidv_df, tfidv_df)
print('\n 유사도 계산 \n', tfidv_df_si)

vect = TfidfVectorizer(max_features=4)
tfvect = vect.fit(docs)
print("\n 출력\n", tfvect)

tfidv_df = pd.DataFrame(tfvect.transform(docs).toarray(), columns=sorted(vect.vocabulary_))
tfidv_df.index = ['문서1', '문서2', '문서3', '문서4']
print("\n 출력\n", tfidv_df)
"""
Surprise 패키지로 학습된 추천 알고리즘을 기반으로
특정 사용자가 아직 평점을 매기지 않은 (관람하지 않은) 영화 중에서
개인 취향에 가장 적절한 영화를 추천
"""
# Surprise를 활용한 개인화 추천
# 'n_epochs': 20, 'n_factors': 50

import pandas as pd
from surprise import Reader, Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import SVD

# 데이터 분리 과정 없이 전체를 학습 데이터로 사용
ratings = pd.read_csv("../dataset/ml-latest-small/ratings.csv")
# 오류 발생
# reader = Reader(rating_scale = (0.5, 5))
# data = Dataset.load_from_file(ratings['userId', 'moviId', 'rating'], reader)
# algo = SVD(n_factor=50, random=0)
# algo.fit(data)


# build_full_trainset() 메서드
# 전체 데이터를 학습 데이터로 이용할 수 있는 방법
from surprise.dataset import DatasetAutoFolds

reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0, 0.5))
data_folds = DatasetAutoFolds(ratings_file='../dataset/ml-latest-small/ratings_noh.csv', reader=reader)
trainset = data_folds.build_full_trainset()

# 학습
algo = SVD(n_epochs=20, n_factors=50, random_state=0)
algo.fit(trainset)

## test
# userId =9 사용자로 지정하여 테스트 진행
movies = pd.read_csv(r"../dataset/ml-latest-small/movies.csv")
movieid = ratings[ratings['userId'] == 9]['movieId']

uid = str(9)
iid = str(42)

pred = algo.predict(uid, iid, verbose=True)
## test



# 개인화 추천

# user id 가 9인 고객이 보지 않았던 전체 영화를 추출한 뒤
# 예측 평점으로 영화를 추천
def get_unseen_surprise(ratings, movies, userId):
    seen_movies = ratings[ratings['userId'] == userId]['movieId'].tolist()
    total_movies = movies['movieId'].tolist()
    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]

    print("평점 매긴 영화 수 :", len(seen_movies), "추천 대상 영화 수 : ",len(unseen_movies), "전체 영화 수 :", len(total_movies))

    return unseen_movies

# unseen_movies = get_unseen_surprise(ratings, movies, 9)


# SVD를 활용해 높은 예측 평점을 가진 순으로 영화를 추천

def recomm_movies_by_surprise(algo, userId, unseen_movied, top_n=10):
    predictions = [algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]

    def sortkey_est(pred):
        return pred.est

    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions = predictions[:top_n]

    top_movie_ids = [int(pred.iid) for pred in top_predictions]
    top_movie_rating = [pred.est for pred in top_predictions]
    top_movie_titles = movies[movies.movieId.isin(top_movie_ids)]['title']
    top_movie_pred = [(id, title, rating) for id, title, rating in zip(top_movie_ids, top_movie_titles, top_movie_rating)]

    return top_movie_pred


# 결과 출력
unseen_movies = get_unseen_surprise(ratings, movies, 9)
top_movies_preds = recomm_movies_by_surprise(algo, 9, unseen_movies, top_n=20)

print("#### Recommendation Top 10 ####")
for top_movie in top_movies_preds:
    print(top_movie[1], ":", top_movie[2])




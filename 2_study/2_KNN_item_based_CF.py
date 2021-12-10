# 아이템 기반 최근접 이웃 협업 필터링
import pandas as pd
import numpy as np

movies = pd.read_csv("../dataset/movielens/movies.csv")
ratings = pd.read_csv("../dataset/movielens/ratings.csv")
print("movies shape : {}".format(movies.shape))
print(movies.head(2))
print("\nratings shape : {}".format(ratings.shape))
print(ratings.head(2))

# 데이터 가공
ratings = ratings[['userId', 'movieId', 'rating']]
ratings_matrix = ratings.pivot_table('rating', index='userId', columns = 'movieId')
print(ratings_matrix.head(5))

ratings_movies = pd.merge(ratings, movies, on ='movieId')
ratings_matrix = ratings_movies.pivot_table('rating', index = 'userId', columns = 'title')
print(ratings_matrix.head())

# fill nan '0'
ratings_matrix = ratings_matrix.fillna(0)
print(ratings_matrix.head())


# 영화간 유사도 추출
ratings_matrix_T = ratings_matrix.transpose()
print("\ntranspose 한거\n", ratings_matrix_T)

from sklearn.metrics.pairwise import cosine_similarity
item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)

item_sim_df = pd.DataFrame(data = item_sim, index = ratings_matrix.columns,
                           columns= ratings_matrix.columns)

print("\n item sim df shape : {}".format(item_sim_df.shape))
print(item_sim_df.head())
# 9064 권에 대한 유사도 계산 데이터 프레임완성


# 제품에 대해 유사도 높은 순서대로 출력해보기
print("\nInception (2010)",item_sim_df['Inception (2010)'].sort_values(ascending=False)[1:6])
# 자기자신 0 제외하고


print("##### 개인화 추천 과정 #####")
# 개인화 추천이 될 수 있도록 변형
# 예측 행렬 만드는 식 -1

def predict_rating(ratings_arr, item_sim_arr):
    ratings_pred = ratings_arr.dot(item_sim_arr)/np.array([np.abs(item_sim_arr).sum(axis=1)])
    return ratings_pred

ratings_pred = predict_rating(ratings_matrix.values, item_sim_df.values)
ratings_pred_matrix = pd.DataFrame(data= ratings_pred, index = ratings_matrix.index,
                                   columns= ratings_matrix.columns)
print("\n userId X title 의 대한 '예측' 평점 결과 \n")
print(ratings_pred_matrix.head())


print("##### MSE 지표로 확인하기 #####")
from sklearn.metrics import mean_squared_error

def get_mse(pred,actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()

    return mean_squared_error(pred, actual)

print("아이템 기반 모든 인접 이웃 MSE = ", get_mse(ratings_pred, ratings_matrix.values))



print("\n##### MSE 더 줄이기 #####\n")



# 예측 행렬 만드는 식 -2
def predict_rating_topsim(ratings_arr, item_sim_arr, n=20):
    pred = np.zeros(ratings_arr.shape)

    for col in range(ratings_arr.shape[1]):
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n-1:-1]]

        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(ratings_arr[row, :][top_n_items].T)
            pred[row, col] /= np.sum(np.abs(item_sim_arr[col, :][top_n_items]))

    return pred

ratings_pred = predict_rating_topsim(ratings_matrix.values, item_sim_df.values, n=20)
print("아이템 기반 인접 TOP-20 이웃 MSE = ", get_mse(ratings_pred, ratings_matrix.values))


ratings_pred_matrix = pd.DataFrame(data=ratings_pred, index= ratings_matrix.index,
                                   columns=ratings_matrix.columns)

print("#2로 만든 예측 평점 행렬\n", ratings_pred_matrix)



# 유저 정보 입력
print("\n##### 유저 정보 반영 #####\n")

user_rating_id = ratings_matrix.loc[9, : ]
print(user_rating_id[ user_rating_id > 0].sort_values(ascending= False)[:10])

# 본 영화 제외시켜주는 함수
def get_unseen_movies(ratings_matrix, userId):
    user_rating = ratings_matrix.loc[userId,:]
    already_seen = user_rating[ user_rating > 0].index.tolist()
    movies_list = ratings_matrix.columns.tolist()
    unseen_list = [ movie for movie in movies_list if movie not in already_seen]
    return unseen_list


def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies

# 유저 9 번에 해당하는 추천 결과

unseen_list = get_unseen_movies(ratings_matrix, 9)
recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)
recomm_movies = pd.DataFrame(data = recomm_movies.values,index = recomm_movies.index,
                             columns=['pred_score'])

print("\nuserId = 9 번에 대한 책 추천 결과\n", recomm_movies)


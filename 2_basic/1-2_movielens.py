# SGD 기반으로 행렬 분해 구현, 이를 통해 사용자에게 영화를 추천해주는 시스템 구현
# RMSE 계산 함수 그대로 사용한다.

from sklearn.metrics import mean_squared_error


def get_rmse(R, P, Q, non_zeros):
    error = 0
    full_pred_matrix = np.dot(P, Q.T)

    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zero = R[x_non_zero_ind, y_non_zero_ind]
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]

    mse = mean_squared_error(R_non_zero, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)

    return rmse

def matrix_factorization(R, K, steps =200, learning_rate = 0.01, r_lambda = 0.01):
    num_users, num_items = R.shape
    np.random.seed(1)
    P = np.random.normal(scale=1./K, size=(num_users, K))
    Q = np.random.normal(scale=1./K, size=(num_items, K))

    break_count = 0

    non_zeros = [(i, j, R[i,j] ) for i in range(num_users) for j in range(num_items) if R[i,j] > 0 ]

    for step in range(steps):
        for i, j, r in non_zeros:
            eij = r - np.dot(P[i, :], Q[j, :].T)

            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])


        rmse = get_rmse(R, P, Q, non_zeros)
        if (step%10)==0:
            print("### iteration step :",step,"rmse : ",rmse)

    return P, Q

if __name__== "__main__":
    import pandas as pd
    import numpy as np

    movies = pd.read_csv("../dataset/movielens/movies.csv")
    ratings = pd.read_csv("../dataset/movielens/ratings.csv")
    ratings = ratings[['userId', 'movieId', 'rating']]

    ratings_matrix = ratings.pivot_table('rating', index='userId', columns='movieId')

    rating_movies = pd.merge(ratings, movies, on ='movieId')

    ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title')

    P, Q = matrix_factorization(ratings_matrix.values, K=50, steps=200, learning_rate=0.01, r_lambda=0.01)
    pred_matrix = np.dot(P, Q.T)


ratings_pred_matrix = pd.DataFrame(data=pred_matrix, index= ratings_matrix.index, columns = ratings_matrix.columns)
print("알고리즘 돌린 결과- 예측 평점 행렬 \n", ratings_pred_matrix.head(3))

def get_unseen_movies(ratings_matrix, userId):
    user_rating = ratings_matrix.loc[userId,:]
    already_seen = user_rating[ user_rating > 0].index.tolist()
    movies_list = ratings_matrix.columns.tolist()
    unseen_list = [ movie for movie in movies_list if movie not in already_seen]
    return unseen_list

def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies

unseen_list = get_unseen_movies(ratings_matrix, 9)
recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)
recomm_movies = pd.DataFrame(data=recomm_movies.values,index=recomm_movies.index,columns=['pred_score'])

print("유저 정보 반영한 최종 예측 결과\n", recomm_movies)

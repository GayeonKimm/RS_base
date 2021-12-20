from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# 10만개
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=25, random_state=52)
print('type -- \n', type(testset))
print('len -- \n', len(testset))
# print('len -- \n', len(trainset))

print('value -- \n', testset)
print('value top5 -- \n', testset[:5])


# latent Factor
# SVD

algo = SVD()
algo.fit(trainset)

# 사용자- 아이템 평점 데이터 셋 전체에 대해서 추천을 예측하는 메서드
# 입력된 데이터 세트에 대해 추천 데이터 셋을 만들어줌
predictions = algo.test(testset)
print(predictions)
# print("Prediction type : ", type(predictions), "size : ", len(predictions))
print(predictions[0])
print(predictions[0].uid)
print(predictions[0].iid)
print(predictions[0].est)
# 다른 표현 방법
# [print(pred.uid, pred.iid, pred.est) for pred in predictions[:3]]



# predict()
uid = str(699)
iid = str(234)
pred = algo.predict(uid, iid)
print(pred)


# 평가
# RMSE, MSE 등의 방법이 사용됨

pred_accuracy = accuracy.rmse(predictions)

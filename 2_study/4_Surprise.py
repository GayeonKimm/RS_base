from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=25, random_state=52)
print('type -- \n', type(testset))
print('len -- \n', len(testset))
print('value -- \n', testset)
print('value top5 -- \n', testset[:5])


# latent Factor
# SVD

algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
print("Prediction type : ", type(predictions), "size : ", len(predictions))
print(predictions)

print(predictions[:5])
print(predictions[:1])
print(predictions[0].uid)
print(predictions[0].iid)
print(predictions[0].est)

[print(pred.uid, pred.iid, pred.est) for pred in predictions[:3]]

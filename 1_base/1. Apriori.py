import mlxtend
import numpy as np
import pandas as pd
data = np.array([
    ['우유', '기저귀', '쥬스'],
    ['양상추', '기저귀', '맥주'],
    ['우유', '양상추', '기저귀', '맥주'],
    ['양상추', '맥주']
])
print('0. 데이터 생성')
print(data)

# Apriori Algorithm

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns = te.columns_)
print('1. dataframe 형태')
print(df)


from mlxtend.frequent_patterns import apriori
ap = apriori(df, min_support=0.5, use_colnames=True)
print('3. apriori 적용 결과')
print(ap)
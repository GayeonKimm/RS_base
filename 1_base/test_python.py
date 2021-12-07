# method1
i = 0
for letter in ['a','b','c']:
    print(i, letter )
    i += 1

# method2
letters = ['a','b','c']
for i in range(len(letters)):
    letter = letters[i]
    print(i, letter)

# enumerate 함수 사용
print("\n---enumerate 함수 사용---\n")
for entry in enumerate(['a','b','c']):
    print(entry)
# default가 튜플의 형태로 만듬

for i, letter in enumerate(['a','b','c']):
    print(i, letter)

# 시작 인덱스 바꿀수도 있음
for i , letter in enumerate(['a','b','c'], start = 1):
    print(i, letter)
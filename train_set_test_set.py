# 훈련 세트, 테스트 세트 => 기존 데이터를 일부 떼 내어 활용한다
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
               31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
               35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
               10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
               500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
               700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
               7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1] * 35 + [0] * 14
print(fish_data)

kn = KNeighborsClassifier()
print("배열 슬라이싱")

print(fish_data[0:5])
# print(fish_data[:5])

# print(fish_data[44:49])
print(fish_data[44:])

# 훈련세트, 테스트 세트 준비
train_input = fish_data[:35]
train_target = fish_target[:35]
test_input = fish_data[35:]
test_target = fish_target[35:]

kn.fit(train_input, train_target)
print("테스트 결과")
print(kn.score(test_input, test_target))

# 넘파이 연습
print("넘파이 연습")
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
# print(input_arr)
# print(target_arr)
# 샘플 수, 특성 수 출력
print(input_arr.shape)

# 무작위 인덱스 생성
print("무작위 인덱스 생성")
np.random.seed(42)
index = np.arange(49)
print(index)
np.random.shuffle(index)
print(index)

# 배열 인덱싱
print("배열 인덱싱")
print(input_arr[[1, 3]])
new_train_input = input_arr[index[:35]]
new_train_target = target_arr[index[:35]]
print("새로운 훈련 세트가 잘 만들어졌는지 확인")
print(input_arr[13], new_train_input[0])

new_test_input = input_arr[index[35:]]
new_test_target = target_arr[index[35:]]

plt.scatter(new_train_input[:, 0], new_train_input[:, 1])
plt.scatter(new_test_input[:, 0], new_test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.fit(new_train_input, new_train_target)
score = kn.score(new_test_input, new_test_target)
print(score)

# 테스트 데이터 예측 결과
predict = kn.predict(new_test_input)
print("테스트 데이터 예측 결과")
print(predict)
print(new_test_target)

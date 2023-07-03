import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# bream => 도미
# smelt => 빙어
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 산점도 그리기
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 배열 합치기
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
print("배열을 합쳐보자")
print(length)  # println 과 같은 역할을 하는듯 자동으로 carriage return 되는듯
print(weight)

# 사이킷런을 이용하기 위해 배열을 2차원 리스트로 만들어야함
fish_data = [[l, w] for l, w in zip(length, weight)]
fish_target = [1] * 35 + [0] * 14
print("2차원 배열을 만들어보자")
print(fish_data)
print(fish_target)

# k-최근접 이웃 알고리즘을 구현한 클래스 인스턴스 생성
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)  # 훈련 -> 도미를 찾기 위한 기준을 학습시킴
score = kn.score(fish_data, fish_target)  # 모델 평가 -> 잘 훈련되었는가?
print("훈련 평가, 과연 정확도는?")
print(score)

predict1 = kn.predict([[30, 600]])
print("추측 해봐라 이건 무엇이냐?")
print("도미" if predict1[0] == 1 else "빙어")

print("정확도가 1 아래로 내려가는 값을 찾아보자")
for n in range(5, 50):
    kn.n_neighbors = n
    score = kn.score(fish_data, fish_target)
    if score < 1:
        print(n, score)
        break

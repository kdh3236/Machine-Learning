import tensorflow as tf

# 커스텀 Dense 레이어 정의: 모든 입력 뉴런이 모든 출력 뉴런과 완전히 연결(fully connected) 되는 구조
class MyDenseLayer(tf.keras.layers.Layer): # tf.keras.layers.Layer를 부모 클래스로 하는 새로운 클래스 생성
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__() # 부모 클래스의 기능을 가져옴

        # Initialize weights and bias
        self.W = self.add_weight(shape=[input_dim, output_dim], initializer="random_normal")
        self.b = self.add_weight(shape=[output_dim], initializer="zeros")

    def call(self, inputs):
        # Forward propagate the inputs
        z = tf.matmul(inputs, self.W) + self.b  # 행렬 곱셈 후 편향 더하기

        # 활성화 함수 적용 (Sigmoid)
        output = tf.math.sigmoid(z)

        return output

# 기본적인 Dense 레이어 예제 
layer = tf.keras.layers.Dense(units=2) # 출력 차원이 2

# 모델 정의 (얕은 신경망) - layer을 연결 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),  # (뉴런 개수, active function)
    tf.keras.layers.Dense(2, activation="softmax")  # 출력층 (분류 문제)
])

# 모델 정의 (더 깊은 신경망)
model_deep = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")  # 2개의 클래스
])

# 손실 함수 정의

# 크로스 엔트로피 손실 (다중 클래스 분류)
def cross_entropy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

# 평균 제곱 오차 (회귀)
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))) # 평균 계산하는 함

# 경사 하강법 직접 구현
weights = tf.Variable(tf.random.normal([1])) # 학습 가능한 변수 선언
# tf.constant() 는 tf.Variable의 반대로 학습 가능하지 않은  

learning_rate = 0.01  # 학습률

def compute_loss(weights):
    # 임의의 손실 함수 (예제)
    return tf.reduce_sum(tf.square(weights - 5))  # 가중치가 5가 되는 것이 목표

while True:
    # 자동으로 리소스 관리, with 문 종료 시 자동 소멸 / close, watch를 따로 사용할 필요 없음 
    with tf.GradientTape() as g:
        loss = compute_loss(weights)
        gradient = g.gradient(loss, weights)  # 편미분

    weights.assign_sub(learning_rate * gradient)  # 가중치 업데이트
    # assign_sub > x -= (learning_rate * gradient) 를 계산

    # 학습이 충분하면 종료 (임시 조건)
    if loss.numpy() < 0.01:
        break

# 경사 하강법 알고리즘 (Keras 내장 옵티마이저)
optimizers = {
    "SGD": tf.keras.optimizers.SGD(),
    "Adam": tf.keras.optimizers.Adam(),
    "Adadelta": tf.keras.optimizers.Adadelta(),
    "Adagrad": tf.keras.optimizers.Adagrad(),
    "RMSProp": tf.keras.optimizers.RMSprop()
}

# 옵티마이저를 활용한 학습
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)  # SGD 사용

# 가상 데이터
x = tf.random.normal([100, 10])  # 100개의 샘플, 10개의 특성
y = tf.one_hot(tf.random.uniform([100], maxval=2, dtype=tf.int32), depth=2)  # 0 또는 1을 가지는 원-핫 벡터

# 학습 루프
for epoch in range(100):  # 100번 반복 학습
    with tf.GradientTape() as tape:
        prediction = model(x)  # 예측값
        loss = cross_entropy_loss(y, prediction)  # 손실 함수 계산

    grads = tape.gradient(loss, model.trainable_variables)  # 손실에 대한 기울기 계산
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 가중치 업데이트

    if epoch % 10 == 0:  # 10 에포크마다 출력
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# 드롭아웃 (Regularization)
dropout_layer = tf.keras.layers.Dropout(rate=0.5)

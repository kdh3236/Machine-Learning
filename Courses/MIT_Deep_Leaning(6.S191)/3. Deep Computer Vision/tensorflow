# Generate filter map using convolution
tf.keras.layers.COnv2D(filters = d, kernel_size = (h, w), strides = s)

# Non-linearity
tf.keras.activations.* #ReLU 주로 사용

# Pooling
tf.keras.layers.MaxPool2D(
    pool_size = (2, 2),
    strides = 2
)

import tensorflow as tf

def generate_model():
    model = tf.keras.Sequential([
        # first convolutional layer
        tf.keras.layers.Conv2D(32, filter_size=3, activation = 'relu'),
        tf.keras.layers.Maxpool2D(pool_size=2, strides=2),

        # second convolutional layer
        # 두 번째 layer에서 depth 를 늘리는 이유는 더 상위 level의 분석을 할 수 있도록 하기 위해서이다.
        tf.keras.layers.Conv2D(64, filter_size=3, activation = 'relu'),
        tf.keras.layers.Maxpool2D(pool_size=2, strides=2),

        # fully connected classifier
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax') # 10 outputs
    ])
    
    return model

# Semantic segmentation - Upsampling
tf.keras.layers.Conv2DTranspose(
    filters,          # 출력 채널 개수 (업샘플링 후 Feature Map의 Depth)
    kernel_size,      # 필터(커널) 크기 (예: (3,3))
    strides=(1,1),    # 업샘플링 배율 (보통 (2,2) 사용)
    padding='valid',  # 패딩 방식 ('same' 또는 'valid')
    activation=None   # 활성화 함수 (ReLU, Sigmoid 등)
)


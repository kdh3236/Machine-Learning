# Class 이용한 RNN 직접 구현 
my_rnn = RNN()
hidden_state = [0, 0, 0, 0]

sentence = ["I", "love", "recurrent", "neural"]

for word in sentence:
    prediction, hidden_state = my_rnn(word, hidden_state)

# 최종 예측: networks 가 나오면 잘 학습된 신경망!
next_word_prediction = prediction

class MyRNNCell(tf.keras.layers.Layer):
    def __init__(self, rnn_units, input_dim, ouput_dim):
        super(MyRNNCell, self).__init__()

        # Initialize weight matrices
        self.W_xh = self.add_weight([rnn_units, input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_hy = self.add_weight([output_dim, input_dim])

        # Initialize hidden states to zero
        self.h = tf.zeros([rnn_units, 1])

    def call(self, x):
        # Update the hidden state
        self.h = tf.math.tanh(self.W_hh * self.h + self.W_xh * x)

        # Compute the output
        output = self.W_hy * self.h

        return output, self.h

# tensorflow의 RNN 
tf.keras.layers.SimpleRNN(rnn_units) # 위의 class를 구현해놨다.

# tensorflow의 LSTM
tf.keras.layers.LSTM(mnum_units)

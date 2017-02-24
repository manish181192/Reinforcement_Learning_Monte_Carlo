import tensorflow as tf
class value_approximator:

    W = []
    B = []
    h = []
    in_ = []
    def __init__(self, state_size,learning_rate = 0.001, reg = 0.01, no_layers = 1, hidden_layer_size = 128):
        self.l2_loss = tf.constant(0.0)
        self.state_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        self.reward_placeholder = tf.placeholder(dtype=tf.float32, shape=[None])
        self.in_.append(self.state_placeholder)
        for i in range(no_layers):
            self.W.append(tf.Variable(tf.truncated_normal(shape=[state_size,hidden_layer_size]), dtype= tf.float32))
            self.B.append(tf.Variable(tf.truncated_normal(shape=[hidden_layer_size]), dtype= tf.float32))
            self.l2_loss += tf.nn.l2_loss(self.W[i])
            self.l2_loss += tf.nn.l2_loss(self.B[i])

            self.h.append(tf.nn.xw_plus_b(self.in_[i], self.W[i], self.B[i]))
            self.in_.append(self.h[i])

        self.prediction = self.h[no_layers-1]

        ms_error = 0.5 * tf.square(tf.sub(self.reward_placeholder, self.prediction)) + reg * self.l2_loss

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ms_error)

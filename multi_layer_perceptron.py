import tensorflow as tf
class multilayer_perceptron:
    W = []
    B = []
    in_ = []
    out = []
    out_activated = []

    def __init__(self, state_size, no_layers=1, hidden_layer_size=128, reg_L2 = 0.0):
        self.l2_loss = tf.constant(0.0)
        self.state_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        self.reward_placeholder = tf.placeholder(dtype=tf.float32, shape=[None])
        self.dropout_input = tf.placeholder(dtype= tf.float32)

        self.in_.append(self.state_placeholder)
        for i in range(no_layers):
            if i == 0:
                self.W.append(tf.Variable(tf.truncated_normal(shape=[state_size, hidden_layer_size]), dtype=tf.float32))
                self.B.append(tf.Variable(tf.truncated_normal(shape=[hidden_layer_size]), dtype=tf.float32))
            elif i == no_layers - 1:
                self.W.append(tf.Variable(tf.truncated_normal(shape=[hidden_layer_size, 1]), dtype=tf.float32))
                self.B.append(tf.Variable(tf.truncated_normal(shape=[1]), dtype=tf.float32))
            else:
                self.W.append(
                    tf.Variable(tf.truncated_normal(shape=[hidden_layer_size, hidden_layer_size]), dtype=tf.float32))
                self.B.append(tf.Variable(tf.truncated_normal(shape=[hidden_layer_size]), dtype=tf.float32))
            self.l2_loss += tf.nn.l2_loss(self.W[i])
            self.l2_loss += tf.nn.l2_loss(self.B[i])
            self.out.append(tf.nn.xw_plus_b(self.in_[i], self.W[i], self.B[i]))
            if i == 0 or i == no_layers - 1:
                self.dropout_t = tf.constant(1.0)
            else:
                self.dropout_t = self.dropout_input
            self.out_activated.append(tf.nn.relu(tf.nn.dropout(self.out[i], self.dropout_t)))
            self.in_.append(self.out_activated[i])

        self.prediction = self.out_activated[no_layers - 1]
        self.loss = 0.5 * tf.square(tf.sub(self.reward_placeholder, self.prediction)) + reg_L2*self.l2_loss
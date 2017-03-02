import tensorflow as tf
from multi_layer_perceptron import multilayer_perceptron

class value_approximator:
    '''
            Description: VALUE APPROXIMATOR
            Simple multi-layer perceptron for calculation of values for continuous states.

            constructor Input:
                state_size : dimension of state. Eg. BlackJack [current_sum, dealer_show, usable_ace] = 3
                learning_rate : input to optimizer
                keep_prob : for dropout
                reg : regularization parameter
                no_layers : no of hidden layers
                hidden_layer_size : size of each hidden layer

            methods:
                train_value_approximator : to train the approximator
                predict_value : to get value for given state


    '''

    approximator = None

    def __init__(self, state_size, no_layers=1, hidden_layer_size=128, reg_L2 = 0.0):
        self.approximator = multilayer_perceptron(state_size,
                                                  no_layers= no_layers,
                                                  hidden_layer_size= hidden_layer_size,
                                                  reg_L2 = reg_L2)

    def train_value_approximator(self, no_epochs=200, learning_rate=0.001, dropout_keepprob=0.5, reg=0.0):
        with tf.Session() as session:
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(1e-3)
            self.grads_and_vars = self.optimizer.compute_gradients(self.approximator.loss)
            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

            for epoch in range(no_epochs):
                print("EPOCH : "+ str(epoch))
                f_dict = {self.approximator.state_placeholder: None,
                             self.approximator.reward_placeholder: None,
                             self.approximator.dropout_input: None}

                _, step, loss = session.run([self.train_op, self.global_step, self.approximator.loss], feed_dict=f_dict)
                print("LOSS: "+ loss)

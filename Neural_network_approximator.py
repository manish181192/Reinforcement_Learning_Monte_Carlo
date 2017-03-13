import tensorflow as tf
import numpy as np
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
    global_step = None
    optimizer = None
    grads_and_vars = None
    train_op = None
    model_path = "Models/Value_Approximators/model_08_03_2017_A/model.ckpt"
    checkpoint_path = None
    checkpoint_number = 0
    min_loss = 1000

    def __init__(self, state_size, no_layers=2, hidden_layer_size=128, reg_L2 = 0.3):
        self.approximator = multilayer_perceptron(state_size,
                                                  no_layers= no_layers,
                                                  hidden_layer_size= hidden_layer_size,
                                                  reg_L2 = reg_L2)
        self.model_saver = tf.train.Saver()

    def train_value_approximator(self, no_epochs, states, rewards,learning_rate=0.001, keep_prob=0.8):
        with tf.Session() as session:
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.grads_and_vars = self.optimizer.compute_gradients(self.approximator.loss)
            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

            self.train_step = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(self.approximator.loss)
            session.run(tf.global_variables_initializer())

            if self.checkpoint_path is not None:
                self.model_saver.restore(session, self.checkpoint_path)
                print "MODEL RESTORED"
                print "RESUME TRAINING FROM CHECKPOINT NO:"+ str(self.checkpoint_number)

            for epoch in range(no_epochs):
                print("EPOCH : "+ str(epoch))
                f_dict = {self.approximator.state_placeholder: states,
                          self.approximator.reward_placeholder: rewards,
                          self.approximator.dropout_input: keep_prob}

                _, step, loss = session.run([self.train_step, self.global_step, self.approximator.loss], feed_dict=f_dict)
                print("LOSS: "+ str(loss))
                if loss > 1000:
                    session.run(tf.global_variables_initializer())
                    print "Session Reinitialized"
                if (loss>2 and loss<self.min_loss*0.5) or (loss<2 and loss<self.min_loss*0.7):
                    self.checkpoint_path = self.model_saver.save(sess=session, save_path=self.model_path)
                    self.checkpoint_number += 1
                    print("Model saved in", self.checkpoint_path)
                    self.min_loss = loss

    def predict_value_approximator(self, state):
        f_dict = {
            self.approximator.state_placeholder: state,
            self.approximator.reward_placeholder: np.zeros(1),
            self.approximator.dropout_input: 1.0
        }


        with tf.Session() as session:
            if self.checkpoint_path is None:
                print "No Model Saved Yet"
                return
            session.run(tf.global_variables_initializer())
            self.model_saver.restore(session, self.checkpoint_path)
            print "MODEL RESTORED"
            print "RESUME TRAINING FROM CHECKPOINT NO:" + str(self.checkpoint_number)
            return session.run([self.approximator.prediction], feed_dict=f_dict)

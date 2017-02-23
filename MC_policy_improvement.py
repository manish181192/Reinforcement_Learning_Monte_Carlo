import numpy as np
import matplotlib
import plotting
from collections import defaultdict
from time import time
from random import randint
matplotlib.style.use('ggplot')

# bj = BlackjackEnv()

class monte_carlo:
    ### Monte Carlo Policy Improvement with e-greedy ###
    # state_value = defaultdict(float)
    #[state, action] => reward dictionary
    SA_R_dictionary = defaultdict(float)
    #[state] => [action, reward] dictionary
    S_A_dictionary = defaultdict(tuple)
    # [state] => [action] dictionary
    state_action_map = defaultdict(int)

    action_space = None
    environment = None
    policy = None

    def default_policy(self, state):
        has_action = len(self.S_A_dictionary[state])
        if has_action is 0:
            return randint(0,1)
        else:
            return self.S_A_dictionary[state][0]

    def __init__(self, discount, alpha, action_space, environment, policy = None, imporovement_iterations = 10, no_of_episodes = 10000, TIME_STEP_LIMIT = 100):
        self.environment = environment
        self.action_space = action_space
        if policy is None:
            self.policy = self.default_policy
        else:
            self.policy = policy

        start_time = time()
        for improvement_iter in range(imporovement_iterations):
            for i in range(no_of_episodes):
                print("#EPISODE"+str(i))
                state_list = [TIME_STEP_LIMIT]
                state_list[0] = environment.reset()
                reward = np.zeros(TIME_STEP_LIMIT, dtype= int)
                action = np.zeros(TIME_STEP_LIMIT, dtype= int)

                for time_step in range(TIME_STEP_LIMIT):
                    # SAMPLE STATE, REWARD FROM ENV COMPUTE ACTION VALUE ##########################
                    action[time_step] = self.policy(state_list[time_step])
                    state_, reward[time_step], isTerminate, _ = environment.step( action= action[time_step])
                    print("State:" + str(state_list[time_step])+" Action:" + str(action[time_step])+ " REWARD:"+str(reward[time_step]))
                    ##########################################################
                    if isTerminate:
                        # compute G(t) for each time step ##############
                        # G(t) = Rt + d*G(t+1)
                        g = np.zeros(time_step+1)
                        ts = time_step
                        prev_g = 0
                        while ts>=0:
                            g[ts] = reward[ts]+ discount * prev_g
                            prev_g = g[ts]
                            ts = ts-1
                        #################################################

                        # calculate v(s) for every state ################
                        for ts in range(time_step+1):
                            sa_pair = (state_list[ts], action[ts])
                            previous_mean = self.SA_R_dictionary[sa_pair]
                            # if previous_mean!=0:
                            #     print state
                            mc_error = (g[ts] - previous_mean)
                            new_mean_discounted_reward = self.SA_R_dictionary[sa_pair] + alpha*mc_error

                            self.SA_R_dictionary[sa_pair] = new_mean_discounted_reward
                        break
                        ##################################################
                    else:
                        state_list.append(state_)
                ######## END OF EPISODE #####
            ####### END OF EPOCH #######
            # UPDATE POLICY greedy to values
            for sa in self.SA_R_dictionary:
                state = sa[0]
                current_action = sa[1]
                current_reward = self.SA_R_dictionary[sa]
                is_prev_action_reward_pair = len(self.S_A_dictionary[state])
                if is_prev_action_reward_pair!=0:
                    prev_reward = self.S_A_dictionary[state][1]
                    prev_action = self.S_A_dictionary[state][0]
                    if current_reward > prev_reward:
                        action_reward_tuple = (current_action, current_reward)
                        self.S_A_dictionary[state] = action_reward_tuple
                        self.state_action_map[state] = current_action
                else:
                    action_reward_tuple = (current_action, current_reward)
                    self.S_A_dictionary[state] = action_reward_tuple
                    self.state_action_map[state] = current_action
            end_time = time()
            self.elapsed_time = end_time- start_time

    def display_policy_with_rewards(self):
        for state in self.S_A_dictionary:
            print ("STATE: "+str(state)+" ACTION: "+str(self.S_A_dictionary[state][0])+" REWARD: "+str(self.S_A_dictionary[state][1]))

    def display_policy(self):
        for state in self.state_action_map:
            print ("STATE: "+str(state)+" ACTION: "+str(self.state_action_map[state]))

    def time_elapsed(self):
        return self.elapsed_time

    def plot_graph(self):
        plotting.plot_value_function(self.state_action_map, title="100,000 Steps")
# print("No of States Explored: ", len(S_A_dictionary))
# V_10k = state_value
#

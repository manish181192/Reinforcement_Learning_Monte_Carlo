from env_blackjack import BlackjackEnv
import numpy as np
import matplotlib
import plotting
from collections import defaultdict
from time import time
from random import randint
matplotlib.style.use('ggplot')

bj = BlackjackEnv()

### Monte Carlo Policy Improvement with e-greedy ###
imporovement_iterations = 10
no_of_episodes = 10000
TIME_STEP_LIMIT = 100
discount = 0.6
alpha = 0.9
# state_value = defaultdict(float)
action_space= [0,1]
#[state, action] => reward dictionary
SA_R_dictionary = defaultdict(float)
#[state] => [action, reward] dictionary
S_A_dictionary = defaultdict(tuple)
def policy(state):

    has_action = len(S_A_dictionary[state])
    if has_action is 0:
        return randint(0,1)
    else:
        return S_A_dictionary[state][0]

start_time = time()
for improvement_iter in range(imporovement_iterations):
    for i in range(no_of_episodes):
        print("#EPISODE"+str(i))
        state_list = [TIME_STEP_LIMIT]
        state_list[0] = bj.reset()
        reward = np.zeros(TIME_STEP_LIMIT, dtype= int)
        action = np.zeros(TIME_STEP_LIMIT, dtype= int)

        for time_step in range(TIME_STEP_LIMIT):
            # SAMPLE STATE, REWARD FROM ENV COMPUTE ACTION VALUE ##########################
            action[time_step] = policy(state_list[time_step])
            state_, reward[time_step], isTerminate, _ = bj.step( action= action[time_step])
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
                    previous_mean = SA_R_dictionary[sa_pair]
                    # if previous_mean!=0:
                    #     print state
                    mc_error = (g[ts] - previous_mean)
                    new_mean_discounted_reward = SA_R_dictionary[sa_pair] + alpha*mc_error

                    SA_R_dictionary[sa_pair] = new_mean_discounted_reward
                break
                ##################################################
            else:
                state_list.append(state_)
        ######## END OF EPISODE #####
    ####### END OF EPOCH #######
    # UPDATE POLICY greedy to values
    for sa in SA_R_dictionary:
        state = sa[0]
        current_action = sa[1]
        current_reward = SA_R_dictionary[sa]
        is_prev_action_reward_pair = len(S_A_dictionary[state])
        if is_prev_action_reward_pair!=0:
            prev_reward = S_A_dictionary[state][1]
            prev_action = S_A_dictionary[state][0]
            if current_reward > prev_reward:
                action_reward_tuple = (current_action, current_reward)
                S_A_dictionary[state] =action_reward_tuple
        else:
            action_reward_tuple = (current_action, current_reward)
            S_A_dictionary[state] = action_reward_tuple

end_time = time()
for state in S_A_dictionary:
    print ("STATE: "+str(state)+" ACTION: "+str(S_A_dictionary[state][0])+" REWARD: "+str(S_A_dictionary[state][1]))
print("TIME ELAPSED: "+ str(end_time- start_time))
print("No of States Explored: ", len(S_A_dictionary))
# V_10k = state_value
# plotting.plot_value_function(V_10k, title="100,000 Steps")

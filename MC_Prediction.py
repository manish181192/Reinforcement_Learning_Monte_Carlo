from env_blackjack import BlackjackEnv
import numpy as np
import matplotlib
import plotting
from collections import defaultdict

matplotlib.style.use('ggplot')

bj = BlackjackEnv()

def policy(state):
    score, dealer_score, usable_ace = state
    if score >= 20:
        return 0
    else:
        return 1

### Monte Carlo (Every step) ###

no_of_episodes = 100000
TIME_STEP_LIMIT = 50
discount = 0.9
alpha = 0.001
state_value = defaultdict(float)
for i in range(no_of_episodes):
    print("#EPISODE"+str(i))
    state_list = [TIME_STEP_LIMIT]
    state_list[0] = bj.reset()
    reward = np.zeros(TIME_STEP_LIMIT, dtype= int)
    action = np.zeros(TIME_STEP_LIMIT, dtype= int)

    for time_step in range(TIME_STEP_LIMIT):
            action[time_step] = policy(state_list[time_step])
            state_, reward[time_step], isTerminate, _ = bj.step( action= action[time_step])
            state_list.append(state_)
            # print("State:" + str(state_list[time_step])+" Action:" + str(action[time_step])+ " REWARD:"+str(reward[time_step]))

            if isTerminate:
                # calculate G(t) for every time step
                # G(t) = Rt + d*G(t+1)
                g = np.zeros(time_step+1)
                i = time_step
                prev_g = 0
                while i>=0:
                    g[i] = reward[i]+ discount * prev_g
                    i = i-1

                # calculate v(s) for every state
                for i in range(time_step+1):
                    if state_list[i] in state_value:
                        # incremental mean
                        mc_error = (g[i] - state_value[state_list[i]])
                        state_value[state_list[i]] = state_value[state_list[i]] + alpha*mc_error
                    else:
                        state_value[state_list[i]] = g[i]
                break

for state in state_value:
    print("State: " + str(state) + " Value: " + str(state_value[state]))

print("No of States Explored: ", len(state_value))
V_10k = state_value
plotting.plot_value_function(V_10k, title="100,000 Steps")
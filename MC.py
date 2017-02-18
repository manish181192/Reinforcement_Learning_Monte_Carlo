from env_blackjack import BlackjackEnv
import numpy as np
bj = BlackjackEnv()

def policy(state):
    score, dealer_score, usable_ace = state
    if score >= 20:
        return 0
    else:
        return 1

no_of_episodes = 10
TIME_STEP_LIMIT = 50
state_value = {}
for i in range(no_of_episodes):
    print("#########EPISODE"+str(i))
    state_list = []
    state_list[0] = bj.reset()
    reward = np.zeros(TIME_STEP_LIMIT)
    action = np.zeros(TIME_STEP_LIMIT)

    for time_step in range(TIME_STEP_LIMIT):
            action[time_step] = policy(state_list[time_step])
            state_list[time_step+1], reward[time_step], isTerminate, _ = bj.step( action= action[time_step])
            print("State:" + str(state_list[time_step])+" Action:" + str(action[time_step])+ " REWARD:"+str(reward[time_step]))

            if isTerminate:
                # calculate G(t) for every time step

                # calculate v(s) for every state

                break
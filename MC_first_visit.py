from Environment_BLACK_JACK import ACTION,black_jack

def policy(state):
    if state[0] < 12:
        return ACTION.TWIST
    else:
        return ACTION.STICK

bj = black_jack()
no_of_episodes = 500
discount_factor = 0.9
state_value = {}

for i in range(no_of_episodes):
    ########### SAMPLE EPISODES #################
    time_steps = 0
    state_list = []
    action_list = []
    reward_list = []

    while 1:
        if time_steps == 0:
            state_list[0] = bj.deal_hands(no_of_players=1)  # initial state
            action_list[0] = policy(state_list[0])
            time_steps = time_steps + 1
        else:
            current_reward, next_state, isTerminate = bj.get_next(action= action_list[time_steps-1], state= state_list[time_steps-1])
            reward_list[time_steps-1] = current_reward



            if isTerminate == False:
                state_list[time_steps] = next_state
                action_list[time_steps] = policy(next_state)
            else:
                break
            time_steps = time_steps + 1

        # COMPUTE Gt (Discounted Reward for every time)




from env_blackjack import BlackjackEnv
from enum import Enum

bj = BlackjackEnv()

class ACTION(Enum):
    STICK = 0
    TWIST = 1

def policy(state):
    score, dealer_score, usable_ace = state
    if score >= 20:
        return ACTION.STICK
    else:
        return ACTION.TWIST

no_of_episodes = 10
TIME_STEP_LIMIT = 50
for i in range(no_of_episodes):
    state_ = bj.reset()
    for time_steps in range(TIME_STEP_LIMIT):
            print(state_)
            action = policy(state_)
            print(action)
            state, reward, isTerminate, _ = bj.step( action= action)
            print(reward)
            if isTerminate:
                break
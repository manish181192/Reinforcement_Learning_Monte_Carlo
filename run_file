from MC_policy_improvement import monte_carlo
from env_blackjack import BlackjackEnv

blackJack = BlackjackEnv()
action_space = blackJack.action_space

def my_policy(state):
    score, dealer_score, usable_ace = state
    if score >= 19:
        return 0
    else:
        return 1

monte_carlo = monte_carlo(alpha=0.02, discount= 0.9, environment= blackJack, action_space= action_space, policy= my_policy)

monte_carlo.display_state_values()
# monte_carlo.display_policy()
monte_carlo.plot_value_graph()

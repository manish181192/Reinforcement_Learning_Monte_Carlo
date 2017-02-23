from MC_policy_improvement import monte_carlo
from env_blackjack import BlackjackEnv

blackJack = BlackjackEnv()
action_space = blackJack.action_space

######## POLICY EVALUATION #######
def my_policy(state):
    score, dealer_score, usable_ace = state
    if score >= 19:
        return 0
    else:
        return 1

monte_carlo_evaluation = monte_carlo(alpha=0.02, discount= 0.9, environment= blackJack, action_space= action_space, policy= my_policy)

monte_carlo_evaluation.display_state_values()
monte_carlo_evaluation.plot_value_graph()

######## POLICY IMPROVEMENT #######
mc_optimal_policy = monte_carlo(alpha=0.02, discount= 0.9, environment= blackJack, action_space= action_space)
mc_optimal_policy.display_policy()
mc_optimal_policy.plot_action_graph()

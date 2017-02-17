from enum import Enum
from random import randint,random

class ACTION(Enum):
    STICK = 0
    TWIST = 1

class black_jack:
    def __init__(self):
        #State and range of each sub-state
        # self.current_state = [] # LIST :[Current Sum, Dealer show, Useable ace]
        self.s0_range_current_sum = [12,21]
        self.s1_range_dealer_show = [1-10]
        # self.s2_useable_ace = [0,1]

    def deal_hands(self, no_of_players):

        hands = []
        for i in range(no_of_players):
            current_hand_sum = randint(self.s0_range_current_sum[0], self.s0_range_current_sum[1])
            hands[i] = current_hand_sum
        dealer_show_hand = randint(self.s1_range_dealer_show[0], self.s1_range_dealer_show[1])
        ace_prob = random()
        if(ace_prob > 0.75):
            useable_ace = 1
        else:
            useable_ace = 0

        return hands, dealer_show_hand, useable_ace

    #state = [current_sum, Dealer_show, useable_ace]
    #action = [
    # stick => return reward, isTerminate
    # twist => return reward, state, isTerminate
    def get_next(self, action, state):
        if action == ACTION.STICK:
            #deal one card to dealer
            dealer_next_card = randint(self.s1_range_dealer_show[0], self.s1_range_dealer_show[1])
            dealer_sum = state[1]+dealer_next_card
            #calculate reward

            if state[0] > dealer_sum:
                reward = 1
            elif state[0] == dealer_sum:
                reward = 0
            else:
                reward = -1
            return reward,state, True

        if action == ACTION.TWIST:
            #deal next card to player
            next_card = randint(1,10)
            current_sum = state[0]+next_card
            if current_sum > 21:
                reward = -1
                return reward, state, True
            else:
                state[0] = state[0]+next_card
                reward = 0
                return reward, state, False
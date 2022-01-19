from hanabi_learning_environment.rl_env import Agent
import math
import random

# Finds the index of the largest value in a list 
def argmax(llist):
    return llist.index(max(llist))
    
class MyAgent(Agent):
    """Agent that applies a simple heuristic."""


    # This function defines our final chromosome with the list of rules in performance order
    def __init__(self, config, chromosome=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 16, 17, 25, 19, 20, 21, 22, 24, 15, 26, 27], *args, **kwargs):
        """Initialize the agent."""
        self.config = config
        self.chromosome=chromosome
        # Double check that our chromosome is defined as a list
        assert isinstance(chromosome, list)
        self.max_information_tokens = config.get('information_tokens', 8)


    # Function that returns all the unseen cards in the deck by forming a dictionary of all the possible cards
    # And removes discarded cards, cards we can see in other players hands and cards in the fireworks piles 
    def calculate_all_unseen_cards(self, discard_pile, player_hands, fireworks):
        colors = ['Y', 'B', 'W', 'R', 'G']
        full_hanabi_deck=[{"color":c, "rank":r} for c in colors for r in [0,0,0,1,1,2,2,3,3,4]]
        assert len(full_hanabi_deck)==50 # full hanabi deck size.
        result=full_hanabi_deck
        for card in discard_pile:
            if card in result:
                result.remove(card)
        for hand in player_hands[1:]:
            for card in hand:
                if card in result:
                    result.remove(card)
        for (color, height) in fireworks.items():
            for rank in range(height):
                card={"color":color, "rank":rank}
                if card in result:
                    result.remove(card)
        return result             

    # Filters a given set of cards by a hint provided to the function parameter 
    def filter_card_list_by_hint(self, card_list, hint):
        filtered_card_list=card_list
        if hint["color"]!=None:
            filtered_card_list=[c for c in filtered_card_list if c["color"]==hint["color"]]
        if hint["rank"]!=None:
            filtered_card_list=[c for c in filtered_card_list if c["rank"]==hint["rank"]]
        return filtered_card_list


    # Finds out which cards in the card list fit exactly onto the next value of its colour's firework
    def filter_card_list_by_playability(self, card_list, fireworks):
        return [c for c in card_list if self.is_card_playable(c,fireworks)]

    # Finds out which cards in the card list are always going to be unplayable on its colour's firework
    def filter_card_list_by_unplayable(self, card_list, fireworks):
        return [c for c in card_list if c["rank"]<fireworks[c["color"]]]

    # Checks if a particular card fits onto its corresponding firework pile 
    def is_card_playable(self, card, fireworks):
        return card['rank'] == fireworks[card['color']]

    # Invokes every turn for each play but only allow the player at offest 0 (Player whose turn it is)
    # To make an action
    def act(self, observation):
        """Act based on an observation."""
        if observation['current_player_offset'] != 0:
            return None
        fireworks = observation['fireworks']
        card_hints=observation['card_knowledge'][0]
        hand_size=len(card_hints)

        # Builds some useful lists of information about what we hold in our hand and what team-mates know about their hands.
        all_unseen_cards=self.calculate_all_unseen_cards(observation['discard_pile'],observation['observed_hands'],observation['fireworks'])
        possible_cards_by_hand=[self.filter_card_list_by_hint(all_unseen_cards, h) for h in card_hints]
        playable_cards_by_hand=[self.filter_card_list_by_playability(posscards, fireworks) for posscards in possible_cards_by_hand]
        probability_cards_playable=[len(playable_cards_by_hand[index])/len(possible_cards_by_hand[index]) for index in range(hand_size)]
        useless_cards_by_hand=[self.filter_card_list_by_unplayable(posscards, fireworks) for posscards in possible_cards_by_hand]
        probability_cards_useless=[len(useless_cards_by_hand[index])/len(possible_cards_by_hand[index]) for index in range(hand_size)]

        # Apply the first rule that applies to our current state
        
        for rule in self.chromosome:
            # Play any cards that we know are definitely playable on a firework pile
            if rule==0:
                my_hand = observation['observed_hands'][0]
                my_hints = observation['card_knowledge'][0]
                Card_Index = 0
                for card, hint in zip(my_hand, my_hints):
                    if card['color'] != None and self.is_card_playable(card,fireworks):
                        return {'action_type': 'PLAY', 'card_index': Card_Index}
                    Card_Index += 1


            elif rule in [1,2,3,4,5]:
                # Play cards based on set probability thresholds
                if rule==1:
                    threshold=0.9
                elif rule==2:
                    threshold=0.8
                elif rule==3:
                    threshold=0.7
                elif rule==4:
                    threshold=0.6
                else:
                    threshold=0.5
                if max(probability_cards_playable)>threshold:
                    card_index=argmax(probability_cards_playable)
                    return {'action_type': 'PLAY', 'card_index': card_index}


            elif rule in [6, 7, 8, 9, 10]:
                # If lives > 1, play a card based on probability with riskier thresholds as we have lives to spare
                if observation['life_tokens'] > 1:
                    if rule==6:
                        threshold=0.7
                    elif rule==7:
                        threshold=0.65
                    elif rule==8:
                        threshold=0.6
                    elif rule==9:
                        threshold=0.55
                    else:
                        threshold=0.4
                    if max(probability_cards_playable)>threshold:
                        card_index=argmax(probability_cards_playable)
                        return {'action_type': 'PLAY', 'card_index': card_index}


            elif rule==11:
                # Tell other players about unplayable cards they have i.e cards with a lower
                # rank than the lowest firework pile
                if observation['information_tokens'] > 0:
                    for player_offset in range(1, observation['num_players']):
                        Lowest_firework_height = min(observation['fireworks'].values())
                        player_hand = observation['observed_hands'][player_offset]
                        player_hint = observation['card_knowledge'][player_offset]
                        for card, hint in zip(player_hand, player_hint):
                            if card['rank'] < Lowest_firework_height and hint['rank'] is None:
                                return {
                                            'action_type': 'REVEAL_RANK',
                                            'rank': card['rank'],
                                            'target_offset': player_offset
                                            }

            elif rule==12:
                # Tell a piece of information to other players about playable cards they have
                if observation['information_tokens'] > 0:
                        for player_offset in range(1, observation['num_players']):
                            player_hand = observation['observed_hands'][player_offset]
                            player_hints = observation['card_knowledge'][player_offset]
                            for card, hint in zip(player_hand, player_hints):
                                if self.is_card_playable(card,fireworks):
                                    if hint['color'] is None:
                                        return {
                                            'action_type': 'REVEAL_COLOR',
                                            'color': card['color'],
                                            'target_offset': player_offset
                                        }
                                    elif hint['rank'] is None:
                                        return {
                                            'action_type': 'REVEAL_RANK',
                                            'rank': card['rank'],
                                            'target_offset': player_offset
                                            }

            elif rule==13:
                # Tell other players about ones they have in their hand
               if observation['information_tokens'] > 0:
                for player_offset in range(1, observation['num_players']):
                    player_hand = observation['observed_hands'][player_offset]
                    player_hint = observation['card_knowledge'][player_offset]
                    for card, hint in zip(player_hand, player_hint):
                        if card['rank'] == 0 and hint['rank'] is None:
                            return {
                                        'action_type': 'REVEAL_RANK',
                                        'rank': card['rank'],
                                        'target_offset': player_offset
                                        }


            elif rule==14:
                # Tell other players about fives they have in their hand 
               if observation['information_tokens'] > 0:
                for player_offset in range(1, observation['num_players']):
                    player_hand = observation['observed_hands'][player_offset]
                    player_hint = observation['card_knowledge'][player_offset]
                    for card, hint in zip(player_hand, player_hint):
                        if card['rank'] == 4 and hint['rank'] is None:
                            return {
                                        'action_type': 'REVEAL_RANK',
                                        'rank': card['rank'],
                                        'target_offset': player_offset
                                        }

            elif rule == 15:
                # Tell other players the missing information for partially revealed
                # Cards that they are holding 
                if observation['information_tokens'] > 0:
                        for player_offset in range(1, observation['num_players']):
                            player_hand = observation['observed_hands'][player_offset]
                            player_hints = observation['card_knowledge'][player_offset]
                            for card, hint in zip(player_hand, player_hints):
                                if hint['color'] != None and hint['rank'] is None:
                                    return {
                                        'action_type': 'REVEAL_RANK',
                                        'rank': card['rank'],
                                        'target_offset': player_offset
                                    }
                                elif hint['rank'] != None and hint['color'] is None:
                                    return {
                                        'action_type': 'REVEAL_COLOR',
                                        'color': card['color'],
                                        'target_offset': player_offset
                                        }
              
            elif rule == 16:
            # Tell a random player a piece of information about their oldest card
                if observation['information_tokens'] > 0:
                    random_player = random.randint(1, 3)
                    player_hand = observation['observed_hands'][random_player]
                    player_hints = observation['card_knowledge'][random_player]
                    for card, hint in zip(player_hand, player_hints):
                        if hint['color'] == None:
                            return {
                                'action_type': 'REVEAL_COLOR',
                                'color': card['color'],
                                'target_offset': random_player
                                }
                        elif hint['rank'] == None:
                            return {
                                'action_type': 'REVEAL_RANK',
                                'rank': card['rank'],
                                'target_offset': random_player
                                }

           
            elif rule ==  17:
            # Find the most illinformed player (the player with the least number of
            # Overall hints for their hand) and give them some information about their oldest
            # Card
                if observation['information_tokens'] > 0: 	
                    Least_information_count = -10
                    Chosen_player_offset = -10
                    for player_offset in range(1, observation['num_players']):
                        Current_player_information_count = 0
                        player_hints = observation['card_knowledge'][player_offset]   
                        for each_card in player_hints:
                            if each_card["rank"] != None and each_card["color"] != None:
                                Current_player_information_count += 2
                                continue
                            elif each_card["rank"] != None:
                                Current_player_information_count += 1
                                continue
                            elif each_card["color"] != None:
                                Current_player_information_count += 1
                                continue
                        if Current_player_information_count > Least_information_count:
                            Least_information_count = Current_player_information_count
                            Chosen_player_offset = player_offset
                    player_hand = observation['observed_hands'][Chosen_player_offset]
                    player_hints = observation['card_knowledge'][Chosen_player_offset]
                    for card, hint in zip(player_hand, player_hints):
                        if hint['color'] == None:
                            return {
                                'action_type': 'REVEAL_COLOR',
                                'color': card['color'],
                                'target_offset': Chosen_player_offset
                                }
                        elif hint['rank'] == None:
                            return {
                                'action_type': 'REVEAL_RANK',
                                'rank': card['rank'],
                                'target_offset': Chosen_player_offset
                                }
            
            
            elif rule in [18,19,20,21,22]:
                # Discard cards based on set probability thresholds
                if rule==18:
                    threshold=0.9
                elif rule==19:
                    threshold=0.8
                elif rule==20:
                    threshold=0.7
                elif rule==21:
                    threshold=0.6
                else:
                    threshold=0.5
                if observation['information_tokens'] < self.max_information_tokens:
                    if max(probability_cards_useless)>threshold:
                        card_index=argmax(probability_cards_useless)
                        return {'action_type': 'DISCARD', 'card_index': card_index}
            
            
            elif rule==23:
                # Discard any fully revealed cards that we know are discardable 
                my_hand = observation['observed_hands'][0]
                my_hints = observation['card_knowledge'][0]
                Card_Index = 0
                for card, hint in zip(my_hand, my_hints):
                    if card['color'] != None and card['rank'] < fireworks[card['color']]:
                        return {'action_type': 'DISCARD', 'card_index': Card_Index}
                    Card_Index += 1

            
            
            elif rule == 24:
                # Discard a card that is the least likely to affect the highest achievable score for
                # this game i.e. if the lowest pile is 1, then throw away the highest card held for
                # that pile as it's unlikely that pile will score highly
                if observation['information_tokens'] < self.max_information_tokens:
                    Overall_lowest = 6
                    for lowest in observation['fireworks']:
                        if observation['fireworks'][lowest] < Overall_lowest:
                            Overall_lowest = observation['fireworks'][lowest]
                            Lowest_pile = lowest
                    Highest_card_index = 0 
                    Highest_card_least_likely_to_affect_game = 0
                    Card_Index = 0
                    my_hand = observation['observed_hands'][0]
                    my_hints = observation['card_knowledge'][0]
                    for card, hint in zip(my_hand, my_hints):
                        if card['color'] == Lowest_pile and card['rank'] > observation['fireworks'][Lowest_pile]and card['rank'] > Highest_card_least_likely_to_affect_game:
                            Highest_card_least_likely_to_affect_game = card
                            Highest_card_index = Card_Index
                        Card_Index += 1
                    return {'action_type': 'DISCARD', 'card_index': Highest_card_index}
            
                    
            elif rule == 25:
                # Discard our oldest card that has no hints given
                if observation['information_tokens'] < self.max_information_tokens:
                    my_hints = observation['card_knowledge'][0]
                    Card_Index = 0
                    for each_hint in my_hints:
                        if each_hint['color'] == None and each_hint['rank'] == None:
                            return {'action_type': 'DISCARD', 'card_index': Card_Index}
                        Card_Index += 1
                            
            
            elif rule==26:
                # Discard our oldest card
                if observation['information_tokens'] < self.max_information_tokens:
                    return {'action_type': 'DISCARD', 'card_index': 0}


            elif rule==27:
                # Play the card with the highest probability of being playable 
                return {'action_type': 'PLAY', 'card_index': argmax(probability_cards_playable)}


            else:
                raise Exception("Rule not defined: "+str(rule))
        # If no rule has fired, then raise an exception (shouldn't happen provided rule 26, 27 are present
        raise Exception("No rule fired for game situation - faulty rule set")

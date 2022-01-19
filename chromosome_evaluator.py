import sys
from hanabi_learning_environment import rl_env
from rule_agent_chromosome import RuleAgentChromosome
import os, contextlib
import random

# This function simulates a full Hanabi game based on the chromosome provided to
# the function parameter, and plays as many games as specified.
# It returns the average score over all the game that have been simulated
def run(num_episodes, num_players, chromosome, verbose=False):
    """Run episodes."""
    environment = rl_env.make('Hanabi-Full', num_players=num_players)
    game_scores = []
    for episode in range(num_episodes):
        observations = environment.reset()
        agents = [RuleAgentChromosome({'players': num_players}, chromosome) for _ in range(num_players)]
        done = False
        episode_reward = 0
        while not done:
            for agent_id, agent in enumerate(agents):
                observation = observations['player_observations'][agent_id]
                action = agent.act(observation)
                if observation['current_player'] == agent_id:
                    assert action is not None
                    current_player_action = action
                    if verbose:
                        print("Player", agent_id, "to play")
                        print("Player", agent_id, "View of cards", observation["observed_hands"])
                        print("Fireworks", observation["fireworks"])
                        print("Player", agent_id, "chose action", action)
                        print()
                else:
                    assert action is None
            # Make an environment step.
            observations, reward, done, unused_info = environment.step(current_player_action)
            if reward < 0:
                reward = 0  
            episode_reward += reward
        if verbose:
            print("Game over.  Fireworks", observation["fireworks"], "Score=", episode_reward)
        game_scores.append(episode_reward)
    return sum(game_scores) / len(game_scores)


if __name__ == "__main__":
    num_players = 4
    # This is our starting chromosome used as a baseline for the GA to work with
    chromosome = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 16, 17, 25, 19, 20, 21, 22, 24, 15, 26, 27]
    result = run(1, num_players, chromosome)
    Fitness_List = [result]

    # For 1000 chromosome generations, find a number between 1 and 3 and execute
    # That particular mutation to the chromosome 
    for x in range(250):
        Choice = random.randint(1, 3)

        
        # Choice 1, takes a list of the available genomes that aren't already in the chromosome
        # and adds a random one. It then simulates a Hanabi game with the mutated chromosome and
        # compares it with the original one, updating it if necessary
        if Choice == 1:
            mutation_sample = [0, 1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
            potential_mutations = []
            for i in mutation_sample:
                if i not in chromosome:
                    potential_mutations.append(i)
            if potential_mutations == []:
                pass
            else:
                mutated_chromosome = chromosome.copy()
                mutated_chromosome.append(random.choice(potential_mutations))
                mutation_result = run(1, num_players, mutated_chromosome)
                if mutation_result > result:
                    Fitness_List[0] = mutation_result
                    chromosome = mutated_chromosome
                    result = mutation_result
                else:
                    pass

        # Choice 2, takes the chromosome and removes a random rule from it making sure not to remove
        # Rule 26 / 27 which are always required. Similarly to above it then simulates a game, compares
        # It with the original and updates our variables as required
        if Choice == 2:
            mutation_choices = []
            if chromosome == [26, 27] or chromosome == [27, 26] or chromosome == []:
                pass
            else:
                for o in chromosome:
                    if o != 26 and o != 27:
                        mutation_choices.append(o)
                    else:
                        pass
                mutation_choice = random.choice(mutation_choices)
                mutated_chromosome = chromosome.copy()
                mutated_chromosome.remove(mutation_choice)
                mutation_result = run(1, num_players, mutated_chromosome)
                if mutation_result > result:
                    Fitness_List[0] = mutation_result
                    chromosome = mutated_chromosome
                    result = mutation_result
                else:
                    pass

        # Choice 3 takes a random two values from the chromosome and swaps their positions
        # It follows with the relevant simulation and comparisons, and updates our best chromosome
        # and fitness score
        if Choice == 3:
            Mutated_Chromosome = chromosome.copy()
            First_sample = random.choice(Mutated_Chromosome)
            Index_One = Mutated_Chromosome.index(First_sample)
            Second_sample = random.choice(Mutated_Chromosome)
            while First_sample == Second_sample:
                Second_sample = random.choice(Mutated_Chromosome)
            Index_Two = Mutated_Chromosome.index(Second_sample)
            Mutated_Chromosome[Index_One], Mutated_Chromosome[Index_Two] = Mutated_Chromosome[Index_Two], Mutated_Chromosome[Index_One]
            mutation_result = run(1, num_players, Mutated_Chromosome)
            if mutation_result > result:
                Fitness_List[0] = mutation_result
                chromosome = Mutated_Chromosome
                result = mutation_result
            else:
                pass
            # Prints the fitness score at each generation so we can see how the choromosome changes
        print("The best fitness score so far is: ", result)
    # Prints our final chromosome 
    print("The Best Chromosome is: ", chromosome)

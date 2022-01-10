# A way to evaluate RuleAgentChromosome
# The objective of this class is that it could easily be extended 
# into a genentic algorithm engine to improve chromosomes.
# M. Fairbank. October 2021.
import sys
from hanabi_learning_environment import rl_env
from rule_agent_chromosome import RuleAgentChromosome
import os, contextlib
import random


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
                reward = 0  # we're changing the rules so that losing all lives does not result in the score being zeroed.
            episode_reward += reward

        if verbose:
            print("Game over.  Fireworks", observation["fireworks"], "Score=", episode_reward)
        game_scores.append(episode_reward)
    return sum(game_scores) / len(game_scores)


if __name__ == "__main__":
    # TODO you could potentially code a genetic algorithm in here...
    num_players = 4
    chromosome = [1, 5, 6, 8, 10, 12, 9, 17, 18]
    result = run(1, num_players, chromosome)
    Fitness_List = [result]
    for x in range(100000):
        Choice = random.randint(1, 3)
        # Insert a number
        if Choice == 1:
            mutation_sample = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
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

        # Delete a number
        if Choice == 2:
            mutation_choices = []
            if chromosome == [17, 18] or chromosome == [18, 17] or chromosome == []:
                pass
            else:
                for o in chromosome:
                    if o != 17 and o != 18:
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

        # Swap 2 of the numbers
        if Choice == 3:
            Mutated_Chromosome = chromosome.copy()
            First_sample = random.choice(Mutated_Chromosome)
            Index_One = Mutated_Chromosome.index(First_sample)
            Second_sample = random.choice(Mutated_Chromosome)
            while First_sample == Second_sample:
                Second_sample = random.choice(Mutated_Chromosome)
            Index_Two = Mutated_Chromosome.index(Second_sample)
            Mutated_Chromosome[Index_One], Mutated_Chromosome[Index_One] = Mutated_Chromosome[Index_Two], Mutated_Chromosome[Index_One]
            mutation_result = run(1, num_players, chromosome)
            if mutation_result > result:
                Fitness_List[0] = mutation_result
                chromosome = Mutated_Chromosome
                result = mutation_result
            else:
                pass
        print(Fitness_List)

    print("The Best Chromosome is: ", chromosome)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 00:58:37 2019

@author: brbonham
"""
import sys
sys.path
sys.path.append("/Users/brbonham/Documents/Reinforcement_Learning/atari-master/")
import gym
import argparse
import numpy as np
import atari_py
from gym_wrappers import MainGymWrapper
import copy

import random
from statistics import mean
import os
import shutil
from game_models.base_game_model import BaseGameModel
from convolutional_neural_network import ConvolutionalNeuralNetwork

FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)


GAMMA = 0.99
MEMORY_SIZE = 900000
BATCH_SIZE = 32
TRAINING_FREQUENCY = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 40000
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 50000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.02
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS

class GEGameModel(BaseGameModel):

    model = None

    def __init__(self, game_name, mode_name, input_shape, action_space, logger_path, model_path):
        BaseGameModel.__init__(self,
                               game_name,
                               mode_name,
                               logger_path,
                               input_shape,
                               action_space)
        self.model_path = model_path
        self.model = ConvolutionalNeuralNetwork(input_shape, action_space).model

    def _predict(self, state):
        if np.random.rand() < 0.02:
            return random.randrange(self.action_space)
        q_values = self.model.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0])


class GESolver(GEGameModel):

    def __init__(self, game_name, input_shape, action_space):
        testing_model_path = "./output/neural_nets/" + game_name + "/ge/testing/model.h5"
        assert os.path.exists(os.path.dirname(testing_model_path)), "No testing model in: " + str(testing_model_path)
        GEGameModel.__init__(self,
                             game_name,
                             "GE testing",
                             input_shape,
                             action_space,
                             "./output/logs/" + game_name + "/ge/testing/" + self._get_date() + "/",
                             testing_model_path)
        self.model.load_weights(self.model_path)

    def move(self, state):
        return self._predict(state)


class GETrainer(GEGameModel):

    run = 0
    generation = 0
    selection_rate = 0.1
    mutation_rate = 0.01
    population_size = 100
    random_weight_range = 1.0
    parents = int(population_size * selection_rate)

    def __init__(self, game_name, input_shape, action_space):
        GEGameModel.__init__(self,
                             game_name,
                             "GE training",
                             input_shape,
                             action_space,
                             "./output/logs/" + game_name + "/ge/training/"+ self._get_date() + "/",
                             "./output/neural_nets/" + game_name + "/ge/" + self._get_date() + "/model.h5")
        if os.path.exists(os.path.dirname(self.model_path)):
            shutil.rmtree(os.path.dirname(self.model_path), ignore_errors=True)
        os.makedirs(os.path.dirname(self.model_path))

    def move(self, state):
        pass

    def genetic_evolution(self, env):
        print ("population_size: " + str(self.population_size) +\
              ", mutation_rate: " + str(self.mutation_rate) +\
              ", selection_rate: " + str(self.selection_rate) +\
              ", random_weight_range: " + str(self.random_weight_range))
        population = None

        while True:
            print(('{{"metric": "generation", "value": {}}}'.format(self.generation)))

            # 1. Selection
            parents = self._strongest_parents(population, env)

            self._save_model(parents)  # Saving main model based on the current best two chromosomes

            # 2. Crossover (Roulette selection)
            pairs = []
            while len(pairs) != self.population_size:
                pairs.append(self._pair(parents))

            # # 2. Crossover (Rank selection)
            # pairs = self._combinations(parents)
            # random.shuffle(pairs)
            # pairs = pairs[:self.population_size]

            base_offsprings = []
            for pair in pairs:
                offsprings = self._crossover(pair[0][0], pair[1][0])
                base_offsprings.append(offsprings[-1])

            # 3. Mutation
            new_population = self._mutation(base_offsprings)
            population = new_population
            self.generation += 1

    def _pair(self, parents):
        total_parents_score = sum([x[1] for x in parents])
        pick = random.uniform(0, total_parents_score)
        pair = [self._roulette_selection(parents, pick), self._roulette_selection(parents, pick)]
        return pair

    def _roulette_selection(self, parents, pick):
        current = 0
        for parent in parents:
            current += parent[1]
            if current > pick:
                return parent
        return random.choice(parents) # Fallback

    def _combinations(self, parents):
        combinations = []
        for i in range(0, len(parents)):
            for j in range(i, len(parents)):
                combinations.append((parents[i], parents[j]))
        return combinations

    def _strongest_parents(self, population, env):
        if population is None:
            population = self._initial_population()
        scores_for_chromosomes = []
        for i in range(0, len(population)):
            chromosome = population[i]
            scores_for_chromosomes.append((chromosome, self._gameplay_for_chromosome(chromosome, env)))

        scores_for_chromosomes.sort(key=lambda x: x[1])
        top_performers = scores_for_chromosomes[-self.parents:]
        top_scores = [x[1] for x in top_performers]
        print(('{{"metric": "population", "value": {}}}'.format(mean([x[1] for x in scores_for_chromosomes]))))
        print(('{{"metric": "top_min", "value": {}}}'.format(min(top_scores))))
        print(('{{"metric": "top_avg", "value": {}}}'.format(mean(top_scores))))
        print(('{{"metric": "top_max", "value": {}}}'.format(max(top_scores))))
        return top_performers

    def _mutation(self, base_offsprings):
        offsprings = []
        for offspring in base_offsprings:
            offspring_mutation = copy.deepcopy(offspring)

            for a in range(0, len(offspring_mutation)):  # 10
                a_layer = offspring_mutation[a]
                for b in range(0, len(a_layer)):  # 8
                    b_layer = a_layer[b]
                    if not isinstance(b_layer, np.ndarray):
                        if np.random.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate]):
                            offspring_mutation[a][b] = self._random_weight()
                        continue
                    for c in range(0, len(b_layer)):  # 8
                        c_layer = b_layer[c]
                        if not isinstance(c_layer, np.ndarray):
                            if np.random.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate]):
                                offspring_mutation[a][b][c] = self._random_weight()
                            continue
                        for d in range(0, len(c_layer)):  # 4
                            d_layer = c_layer[d]
                            for e in range(0, len(d_layer)):  # 32
                                if np.random.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate]):
                                    offspring_mutation[a][b][c][d][e] = self._random_weight()
            offsprings.append(offspring_mutation)
        return offsprings

    def _crossover(self, x, y):
        offspring_x = x
        offspring_y = y

        for a in range(0, len(offspring_x)):  # 10
            a_layer = offspring_x[a]
            for b in range(0, len(a_layer)):  # 8
                b_layer = a_layer[b]
                if not isinstance(b_layer, np.ndarray):
                    if random.choice([True, False]):
                        offspring_x[a][b] = y[a][b]
                        offspring_y[a][b] = x[a][b]
                    continue
                for c in range(0, len(b_layer)):  # 8
                    c_layer = b_layer[c]
                    if not isinstance(c_layer, np.ndarray):
                        if random.choice([True, False]):
                            offspring_x[a][b][c] = y[a][b][c]
                            offspring_y[a][b][c] = x[a][b][c]
                        continue
                    for d in range(0, len(c_layer)):  # 4
                        d_layer = c_layer[d]
                        for e in range(0, len(d_layer)):  # 32
                            if random.choice([True, False]):
                                offspring_x[a][b][c][d][e] = y[a][b][c][d][e]
                                offspring_y[a][b][c][d][e] = x[a][b][c][d][e]
        return offspring_x, offspring_y

    def _gameplay_for_chromosome(self, chromosome, env):
        self.run += 1
        self.logger.add_run(self.run)

        self.model.set_weights(chromosome)
        state = env.reset()
        score = 0
        while True:
            action = self._predict(state)
            state_next, reward, terminal, info = env.step(action)
            score += np.sign(reward)
            state = state_next
            if terminal:
                self.logger.add_score(score)
                return score

    def _initial_population(self):
        weights = self.model.get_weights()
        chromosomes = []

        for i in range(0, self.population_size):
            chromosome = weights # 1 686 180 params
            for a in range(0, len(weights)): # 10
                a_layer = weights[a]
                for b in range(0, len(a_layer)):  # 8
                    b_layer = a_layer[b]
                    if not isinstance(b_layer, np.ndarray):
                        weights[a][b] = self._random_weight()
                        continue
                    for c in range(0, len(b_layer)):  # 8
                        c_layer = b_layer[c]
                        if not isinstance(c_layer, np.ndarray):
                            weights[a][b][c] = self._random_weight()
                            continue
                        for d in range(0, len(c_layer)):  # 4
                            d_layer = c_layer[d]
                            for e in range(0, len(d_layer)):  # 32
                                weights[a][b][c][d][e] = self._random_weight()
            chromosomes.append(chromosome)
        return chromosomes

    def _random_weight(self):
        return random.uniform(-self.random_weight_range, self.random_weight_range)

    def _save_model(self, parents):
        x = copy.deepcopy(parents[-1][0])
        y = copy.deepcopy(parents[-2][0])
        best_offsprings = self._crossover(x, y)
        self.model.set_weights(best_offsprings[-1])
        self.model.save_weights(self.model_path)


class DDQNGameModel(BaseGameModel):

    def __init__(self, game_name, mode_name, input_shape, action_space, logger_path, model_path):
        BaseGameModel.__init__(self, game_name,
                               mode_name,
                               logger_path,
                               input_shape,
                               action_space)
        self.model_path = model_path
        self.ddqn = ConvolutionalNeuralNetwork(self.input_shape, action_space).model
        if os.path.isfile(self.model_path):
            self.ddqn.load_weights(self.model_path)

    def _save_model(self):
        self.ddqn.save_weights(self.model_path)


class DDQNSolver(DDQNGameModel):

    def __init__(self, game_name, input_shape, action_space):
        testing_model_path = "./output/neural_nets/" + game_name + "/ddqn/testing/model.h5"
        assert os.path.exists(os.path.dirname(testing_model_path)), "No testing model in: " + str(testing_model_path)
        DDQNGameModel.__init__(self,
                               game_name,
                               "DDQN testing",
                               input_shape,
                               action_space,
                               "./output/logs/" + game_name + "/ddqn/testing/" + self._get_date() + "/",
                               testing_model_path)

    def move(self, state):
        if np.random.rand() < EXPLORATION_TEST:
            return random.randrange(self.action_space)
        q_values = self.ddqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0])


class DDQNTrainer(DDQNGameModel):

    def __init__(self, game_name, input_shape, action_space):
        DDQNGameModel.__init__(self,
                               game_name,
                               "DDQN training",
                               input_shape,
                               action_space,
                               "./output/logs/" + game_name + "/ddqn/training/" + self._get_date() + "/",
                               "./output/neural_nets/" + game_name + "/ddqn/" + self._get_date() + "/model.h5")

        if os.path.exists(os.path.dirname(self.model_path)):
            shutil.rmtree(os.path.dirname(self.model_path), ignore_errors=True)
        os.makedirs(os.path.dirname(self.model_path))

        self.ddqn_target = ConvolutionalNeuralNetwork(self.input_shape, action_space).model
        self._reset_target_network()
        self.epsilon = EXPLORATION_MAX
        self.memory = []

    def move(self, state):
        if np.random.rand() < self.epsilon or len(self.memory) < REPLAY_START_SIZE:
            return random.randrange(self.action_space)
        q_values = self.ddqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0])

    def remember(self, current_state, action, reward, next_state, terminal):
        self.memory.append({"current_state": current_state,
                            "action": action,
                            "reward": reward,
                            "next_state": next_state,
                            "terminal": terminal})
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)

    def step_update(self, total_step):
        if len(self.memory) < REPLAY_START_SIZE:
            return

        if total_step % TRAINING_FREQUENCY == 0:
            loss, accuracy, average_max_q = self._train()
            self.logger.add_loss(loss)
            self.logger.add_accuracy(accuracy)
            self.logger.add_q(average_max_q)

        self._update_epsilon()

        if total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0:
            self._save_model()

        if total_step % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            self._reset_target_network()
            print(('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon)))
            print(('{{"metric": "total_step", "value": {}}}'.format(total_step)))

    def _train(self):
        batch = np.asarray(random.sample(self.memory, BATCH_SIZE))
        if len(batch) < BATCH_SIZE:
            return

        current_states = []
        q_values = []
        max_q_values = []

        for entry in batch:
            current_state = np.expand_dims(np.asarray(entry["current_state"]).astype(np.float64), axis=0)
            current_states.append(current_state)
            next_state = np.expand_dims(np.asarray(entry["next_state"]).astype(np.float64), axis=0)
            next_state_prediction = self.ddqn_target.predict(next_state).ravel()
            next_q_value = np.max(next_state_prediction)
            q = list(self.ddqn.predict(current_state)[0])
            if entry["terminal"]:
                q[entry["action"]] = entry["reward"]
            else:
                q[entry["action"]] = entry["reward"] + GAMMA * next_q_value
            q_values.append(q)
            max_q_values.append(np.max(q))

        fit = self.ddqn.fit(np.asarray(current_states).squeeze(),
                            np.asarray(q_values).squeeze(),
                            batch_size=BATCH_SIZE,
                            verbose=0)
        loss = fit.history["loss"][0]
        accuracy = fit.history["acc"][0]
        return loss, accuracy, mean(max_q_values)

    def _update_epsilon(self):
        self.epsilon -= EXPLORATION_DECAY
        self.epsilon = max(EXPLORATION_MIN, self.epsilon)

    def _reset_target_network(self):
        self.ddqn_target.set_weights(self.ddqn.get_weights())


class GEGameModel(BaseGameModel):

    model = None

    def __init__(self, game_name, mode_name, input_shape, action_space, logger_path, model_path):
        BaseGameModel.__init__(self,
                               game_name,
                               mode_name,
                               logger_path,
                               input_shape,
                               action_space)
        self.model_path = model_path
        self.model = ConvolutionalNeuralNetwork(input_shape, action_space).model

    def _predict(self, state):
        if np.random.rand() < 0.02:
            return random.randrange(self.action_space)
        q_values = self.model.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0])



class Atari:

    def __init__(self):
        game_name, game_mode, render, total_step_limit, total_run_limit, clip = self._args()
        env_name = game_name + "Deterministic-v4"  # Handles frame skipping (4) at every iteration
        env = MainGymWrapper.wrap(gym.make(env_name))
        self._main_loop(self._game_model(game_mode, game_name, env.action_space.n), env, render, total_step_limit, total_run_limit, clip)

    def _main_loop(self, game_model, env, render, total_step_limit, total_run_limit, clip):
        if isinstance(game_model, GETrainer):
            game_model.genetic_evolution(env)

        run = 0
        total_step = 0
        while True:
            if total_run_limit is not None and run >= total_run_limit:
                print ("Reached total run limit of: " + str(total_run_limit))
                exit(0)

            run += 1
            current_state = env.reset()
            step = 0
            score = 0
            while True:
                if total_step >= total_step_limit:
                    print ("Reached total step limit of: " + str(total_step_limit))
                    exit(0)
                total_step += 1
                step += 1

                if render:
                    env.render()

                action = game_model.move(current_state)
                next_state, reward, terminal, info = env.step(action)
                if clip:
                    np.sign(reward)
                score += reward
                game_model.remember(current_state, action, reward, next_state, terminal)
                current_state = next_state

                game_model.step_update(total_step)

                if terminal:
                    game_model.save_run(score, step, run)
                    break

    def _args(self):
        parser = argparse.ArgumentParser()
        available_games = list((''.join(x.capitalize() or '_' for x in word.split('_')) for word in atari_py.list_games()))
        parser.add_argument("-g", "--game", help="Choose from available games: " + str(available_games) + ". Default is 'breakout'.", default="Breakout")
        parser.add_argument("-m", "--mode", help="Choose from available modes: ddqn_train, ddqn_test, ge_train, ge_test. Default is 'ddqn_training'.", default="ddqn_training")
        parser.add_argument("-r", "--render", help="Choose if the game should be rendered. Default is 'False'.", default=False, type=bool)
        parser.add_argument("-tsl", "--total_step_limit", help="Choose how many total steps (frames visible by agent) should be performed. Default is '5000000'.", default=5000000, type=int)
        parser.add_argument("-trl", "--total_run_limit", help="Choose after how many runs we should stop. Default is None (no limit).", default=None, type=int)
        parser.add_argument("-c", "--clip", help="Choose whether we should clip rewards to (0, 1) range. Default is 'True'", default=True, type=bool)
        args = parser.parse_args()
        game_mode = args.mode
        game_name = args.game
        render = args.render
        total_step_limit = args.total_step_limit
        total_run_limit = args.total_run_limit
        clip = args.clip
        print ("Selected game: " + str(game_name))
        print ("Selected mode: " + str(game_mode))
        print ("Should render: " + str(render))
        print ("Should clip: " + str(clip))
        print ("Total step limit: " + str(total_step_limit))
        print ("Total run limit: " + str(total_run_limit))
        return game_name, game_mode, render, total_step_limit, total_run_limit, clip

    def _game_model(self, game_mode,game_name, action_space):
        if game_mode == "ddqn_training":
            return DDQNTrainer(game_name, INPUT_SHAPE, action_space)
        elif game_mode == "ddqn_testing":
            return DDQNSolver(game_name, INPUT_SHAPE, action_space)
        elif game_mode == "ge_training":
            return GETrainer(game_name, INPUT_SHAPE, action_space)
        elif game_mode == "ge_testing":
            return GESolver(game_name, INPUT_SHAPE, action_space)
        else:
            print ("Unrecognized mode. Use --help")
            exit(1)


if __name__ == "__main__":
    Atari()

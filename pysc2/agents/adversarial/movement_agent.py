# for move to beacon

import numpy as np
from pysc2.lib import features, actions

from base_agent import BaseNeatAgent

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


class MovementAgent(BaseNeatAgent):
    def __init__(self, **kwargs):
        super(MovementAgent, self).__init__(**kwargs)
        self.id = kwargs.get('id', 0)
        self.old_distance = None
        self.distance_to_beacon = 0

    def fitness_calculation_setup(self, obs):
        pass

    def reinitialize(self):
        self.fitness = 0
        self.target = [0] * 2
        self.obs = None
        self.latest_position = [0] * 2
        self.nn_output = []
        self.first_move = True
        self.step_counter = 1
        self.old_distance = None
        self.distance_to_beacon = 0

    def retrieve_handcrafted_inputs(self, obs):
        """[player_x, player_y, beacon_x, beacon _y, distance_to_beacon]"""
        player_position = self.get_current_location(obs)

        beacon_y, beacon_x = (obs.observation.feature_screen.player_relative ==
                              features.PlayerRelative.NEUTRAL).nonzero()
        beacon = list(zip(beacon_x, beacon_y))
        beacon_center = np.mean(beacon, axis=0).round()

        normalized_inputs = [player_position[0] / self.max_map_width,
                             player_position[1] / self.max_map_height,
                             beacon_center[0] / self.max_map_width,
                             beacon_center[1] / self.max_map_height
                             ]

        distance = self.retrieve_distance_to_beacon(obs)
        normalized_inputs.append(distance / np.hypot(self.max_map_height, self.max_map_width))

        return normalized_inputs

    def step(self, obs):
        self.obs = obs
        self.step_counter += 1

        if self.can_do_action(obs, actions.FUNCTIONS.Move_screen.id):
            displacement = [self.nn_output[0], self.nn_output[1]]
            move = self.move_unit(obs)
            if move["status"] is "ARRIVED_AT_TARGET":
                step_size = 10
                self.movement_step(step_size, displacement, obs)
                return actions.FUNCTIONS.Move_screen("now", (self.target[0], self.target[1]))
            else:
                return move["function"]
        if self.can_do_action(obs, actions.FUNCTIONS.select_army.id):
            return actions.FUNCTIONS.select_army("select")

        return actions.FUNCTIONS.no_op()

    def calculate_fitness(self, obs):
        # basic reward
        # self.fitness += obs.reward
        # return self.fitness

        # distance reward
        if self.old_distance is not None:
            self.old_distance = self.distance_to_beacon

        self.distance_to_beacon = self.retrieve_distance_to_beacon(obs)

        if self.old_distance is None:
            self.old_distance = self.distance_to_beacon

        distance_weight = 2 / self.step_counter
        beacon_weight = 5
        rew = (self.old_distance - self.distance_to_beacon) / (200 ** 0.5)
        rew = max(0, rew)
        self.fitness = self.fitness + (distance_weight * rew) + (beacon_weight * obs.reward)
        print(self.fitness)
        return self.fitness

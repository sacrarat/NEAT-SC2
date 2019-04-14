"""
base agent for neat training
for customised behaviour just need to extend this agent and override the step, calculate fitness and the input
retrieval functions
some inspiration from https://github.com/skjb/pysc2-tutorial and https://github.com/alexmirov95/RunAwAI
"""

import numpy as np
from pysc2.lib import actions, features

from pysc2.agents import base_agent

# to identify units fromm feature layers
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


class BaseNeatAgent(base_agent.BaseAgent):
    """
    extends the inbuilt pysc2 base agent. provides additional helper functions that may be useful for concrete
    implementations
    """

    def __init__(self, **kwargs):
        super(BaseNeatAgent, self).__init__()
        self.max_map_width = kwargs.get('map_width', 32)
        self.max_map_height = kwargs.get('map_height', 20)
        # self.max_map_width = 32
        # self.max_map_height = 20  # hardcoded by manual checking for the defeat roaches map template
        self.target = [0] * 2  # target movement location for units
        self.obs = None
        self.latest_position = [0] * 2  # tracks the latest position for movement
        self.nn_output = []  # output of the neural network used in the step function
        self.fitness = 0
        self.first_move = True
        self.step_counter = 1
        self.train = False  # whether agent is being used to train
        self.max_fitness = 0  # max fitness obtainable for the episode
        self.genome_threshold = 0  # threshold used to gauge winning performance

    def reinitialize(self):
        """reset values for an episode"""
        self.fitness = 0
        self.target = [0] * 2
        self.obs = None
        self.latest_position = [0] * 2
        self.nn_output = []
        self.first_move = True
        self.step_counter = 1
        self.max_fitness = 0
        self.genome_threshold = 0

    def move_unit(self, obs):
        """
        movement function that can be used for directing movement in a way such that no new movement command is given
        if the past movement command isnt complete
        :param obs: pysc2 observation
        :return: dict - function is pysc2 action and status denotes whether reached destination target
        """
        target = self.target
        current_location = self.get_current_location(obs)
        distance_to_target = 2
        if ((abs(current_location[0] - target[0]) >= distance_to_target) or
                (abs(current_location[1] - target[1]) >= distance_to_target)):
            return {
                "function": actions.FUNCTIONS.Move_screen("now", (self.target[0], self.target[1])),
                "status": "MOVING_TO_TARGET"
            }
        else:
            return {
                "function": actions.FUNCTIONS.Move_screen("now", (self.target[0], self.target[1])),
                "status": "ARRIVED_AT_TARGET"
            }

    @staticmethod
    def can_do_action(obs, action):
        """checks if the passed pysc2 action is available to do"""
        return action in obs.observation.available_actions

    def xy_locs(self, mask):
        """get xy locations . Mask should be a set of bools from comparison with a feature layer."""
        y, x = mask.nonzero()
        return list(zip(x, y))

    @staticmethod
    def calculate_hitpoints(features, unit_alliance):
        """
        calculate total hitpoints (health + shield) for self or enemy
        :param features: feature units to extract info from
        :param unit_alliance:
        :return: features.PlayerRelative.SELF or features.PlayerRelative.ENEMY
        """
        hitpoints = 0
        units = [unit for unit in features if unit.alliance == unit_alliance]
        for unit in units:
            hitpoints += unit.health + unit.shield
        return hitpoints

    def get_current_location(self, obs):
        """returns [xcoord, ycoord] of current player location"""
        player_relative = obs.observation.feature_screen.player_relative
        player_y, player_x = (
                player_relative == features.PlayerRelative.SELF).nonzero()
        if player_x.size == 0:  # did this because move to beacon would crash
            return self.latest_position
        else:
            self.latest_position = [int(player_x.mean()), int(player_y.mean())]
            return self.latest_position

    def set_target_destination(self, coordinates):
        """Sets the target destination to an x and y tuple in coordinates"""
        x = coordinates[0]
        y = coordinates[1]
        if x > self.max_map_width:
            # print("Invalid target coordinates.")
            x = self.max_map_width
        if x < 0:
            # print("Invalid target coordinates.")
            x = 0
        if y > self.max_map_height:
            # print("Invalid target coordinates.")
            y = self.max_map_height
        if y < 0:
            # print("Invalid target coordinates.")
            y = 0
        self.target = [x, y]

    def movement_step(self, distance, displacement, obs):
        """Will have selected unit(s) move a specified distance in the
        directions cardinal directions, diagonals, and stay put"""
        new_target = self.get_current_location(obs)

        new_target[0] += distance * displacement[0]
        new_target[1] += distance * displacement[1]

        # cap map bounds of new target within map dimensions
        border_limit = 2  # target will not be set within border_limit distance of the edge of map
        if new_target[0] >= (self.max_map_height - border_limit):
            new_target[0] = (self.max_map_height - border_limit)
        if new_target[1] >= (self.max_map_width - border_limit):
            new_target[1] = (self.max_map_width - border_limit)
        if new_target[0] <= border_limit:
            new_target[0] = border_limit
        if new_target[1] <= border_limit:
            new_target[1] = border_limit

        self.set_target_destination(new_target)

    def calculate_fitness(self, obs):
        return self.fitness + obs.reward

    def step(self, obs):
        """default behaviour returns no ops"""
        super(BaseNeatAgent, self).step(obs)
        self.obs = obs

        return actions.FUNCTIONS.no_op()

    def retrieve_inputs(self, obs, input_type):
        if input_type == self.FeatureInputType.Handcrafted:
            return self.retrieve_handcrafted_inputs(obs)
        elif input_type == self.FeatureInputType.Pixel:
            return self.retrieve_pixel_inputs(obs)

    @staticmethod
    def retrieve_pixel_inputs(obs):
        """just extracts one layer from feature layers"""
        def preprocess_screen(feature_screen):
            # extract player id layer from feature layers for input to neural network
            layers = []
            assert feature_screen.shape[0] == len(features.SCREEN_FEATURES)
            for i in range(len(features.SCREEN_FEATURES)):
                if i == features.SCREEN_FEATURES.player_id.index:
                    layers.append(
                        feature_screen[i:i + 1] / features.SCREEN_FEATURES[i].scale)
            return np.array(layers).flatten()

        screen = np.array(obs.observation.feature_screen, dtype=np.float32)
        screen = preprocess_screen(screen)

        return screen

    def retrieve_handcrafted_inputs(self, obs):
        """default gets [player_x, player_y, beacon_x, beacon _y]"""

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

        return normalized_inputs

    def retrieve_distance_to_beacon(self, obs):
        player_position = self.get_current_location(obs)

        beacon_y, beacon_x = (obs.observation.feature_screen.player_relative ==
                              features.PlayerRelative.NEUTRAL).nonzero()
        beacon = list(zip(beacon_x, beacon_y))
        beacon_center = np.mean(beacon, axis=0).round()

        # euclidean distance
        distances = np.linalg.norm(np.array(beacon_center) - player_position)

        return distances.min()

    @staticmethod
    def retrieve_distance_between_positions(pos_1, pos_2):
        distances = np.linalg.norm(np.array(pos_1) - np.array(pos_2))

        return distances.min()

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    def count_units(self, obs):
        units = {}
        feature_units = obs.observation.feature_units
        allies = [unit for unit in feature_units if unit.alliance == _PLAYER_SELF]
        enemies = [unit for unit in feature_units if unit.alliance == _PLAYER_ENEMY]

        units['self'] = len(allies)
        units['enemy'] = len(enemies)

        return units

    class FeatureInputType:
        Handcrafted = 0
        Pixel = 1

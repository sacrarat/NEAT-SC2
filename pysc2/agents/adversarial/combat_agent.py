# several extensions of the base agent for adversarial scenarios
# main difference is in input calculation, fitness calculation, and step functions
# each agent may have extended helper functions for extended calculations
# combat agent 4 and refined agent are successful versions that learn kiting
# all agents currently handle with homogenous compositions of self and enemy
# a hetero agent is implemented but no successful NEAT training results have been obtained so far

import random

import numpy as np
from pysc2.lib import actions, features

from base_agent import BaseNeatAgent
from units import unit_sizes, weapon_ranges, unit_type, unit_speed, unit_health

# to identify units fromm feature layers
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


class CombatAgent(BaseNeatAgent):
    def __init__(self, **kwargs):
        super(CombatAgent, self).__init__(**kwargs)
        self.id = kwargs.get('id', 0)
        self.initial_self_hit_points = 0
        self.initial_enemy_hit_points = 0

    def fitness_calculation_setup(self, obs):
        feature_units = obs.observation.feature_units
        self.initial_self_hit_points = self.calculate_hitpoints(
            feature_units, _PLAYER_SELF)
        self.initial_enemy_hit_points = self.calculate_hitpoints(
            feature_units, _PLAYER_ENEMY)

    @staticmethod
    def retrieve_enemy_location(obs):
        enemy_y, enemy_x = (
                obs.observation.feature_screen.player_relative == _PLAYER_ENEMY).nonzero()
        enemy = list(zip(enemy_x, enemy_y))
        enemy_center = np.mean(enemy, axis=0).round()

        return enemy_center

    @staticmethod
    def retrieve_distance_to_enemy(obs, pos):
        enemy_center = CombatAgent.retrieve_enemy_location(obs)

        distances = np.linalg.norm(np.array(enemy_center) - pos)

        return distances.min()

    def retrieve_handcrafted_inputs(self, obs):
        """
        :param obs: pysc2 observation
        :return:
        [
        current_hp,
         weapon_cooldown,
         enemy_in_range
         ]
        """
        feature_units = obs.observation.feature_units

        ally = [unit for unit in feature_units if unit.alliance == _PLAYER_SELF]
        if len(ally) > 0:
            current_hp = ally[0].health
            current_hp = current_hp / self.initial_self_hit_points
            weapon_cooldown = ally[0].weapon_cooldown
            if weapon_cooldown > 0:
                weapon_cooldown = 1
            else:
                weapon_cooldown = 0
            # dont know why radius is retrieved as zero - pysc2 bug, hardcoded for now
            # radius = ally[0].radius
            radius = 2.5
        else:
            current_hp = 0
            weapon_cooldown = 0
            radius = 5

        weapon_range = 5

        if self.retrieve_distance_to_enemy(obs, self.get_current_location(obs)) < (weapon_range + 2 * radius):
            enemy_in_range = 1
        else:
            enemy_in_range = 0

        return [current_hp, weapon_cooldown, enemy_in_range]

    def movement_step(self, direction, distance, obs):
        new_target = self.get_current_location(obs)

        if direction is "NORTH":
            new_target[1] += distance
        elif direction is "EAST":
            new_target[0] += distance
        elif direction is "SOUTH":
            new_target[1] -= distance
        elif direction is "WEST":
            new_target[0] -= distance
        elif direction is "NORTHEAST":
            new_target[0] += distance
            new_target[1] += distance
        elif direction is "SOUTHEAST":
            new_target[0] += distance
            new_target[1] -= distance
        elif direction is "SOUTHWEST":
            new_target[0] -= distance
            new_target[1] -= distance
        elif direction is "NORTHWEST":
            new_target[0] -= distance
            new_target[1] += distance
        elif direction is "STAY":
            # no movement from current location
            new_target = new_target
        else:
            print("Invalid Direction:", direction)

        # cap map bounds of new target within map dimensions
        border_limit = 1  # target will not be set within borderLimit distance of the edge of map
        if new_target[0] >= (self.max_map_height - border_limit):
            new_target[0] = (self.max_map_height - border_limit)
        if new_target[1] >= (self.max_map_width - border_limit):
            new_target[1] = (self.max_map_width - border_limit)
        if new_target[0] <= border_limit:
            new_target[0] = border_limit
        if new_target[1] <= border_limit:
            new_target[1] = border_limit

        return new_target

    def step(self, obs):
        """network only controls fight or flee decision. flee logic is hardcoded"""
        self.obs = obs
        decision = self.nn_output[0]

        if self.first_move:
            if self.can_do_action(obs, actions.FUNCTIONS.Move_screen.id):
                self.first_move = False
                self.set_target_destination(self.retrieve_enemy_location(obs))
                return self.move_unit(obs)["function"]

        if decision > 0.5:  # fight
            if self.can_do_action(obs, actions.FUNCTIONS.Attack_screen.id):
                player_relative = obs.observation.feature_screen.player_relative
                enemy = self.xy_locs(player_relative == _PLAYER_ENEMY)
                if not enemy:
                    return actions.FUNCTIONS.no_op()

                target = enemy[np.argmax(np.array(enemy)[:, 1])]
                return actions.FUNCTIONS.Attack_screen("now", target)
        else:  # flee
            if self.can_do_action(obs, actions.FUNCTIONS.Move_screen.id):
                move = self.move_unit(obs)
                if move["status"] is "ARRIVED_AT_TARGET":
                    """
                    work out distances in all possible cardinal directions and move to the one with the minimum 
                    distance to enemy while still being outside range
                    """

                    movement_direction_action_space = ["NORTH", "SOUTH", "EAST", "WEST", "NORTHEAST", "SOUTHEAST",
                                                       "SOUTHWEST",
                                                       "NORTHWEST"]
                    positions = []
                    enemy_range = 1
                    movement_step = 4
                    for direction in movement_direction_action_space:
                        positions.append((direction, self.movement_step(
                            direction, enemy_range + movement_step, obs)))
                    distances = []
                    for position in positions:
                        distances.append(
                            (position[0], self.retrieve_distance_to_enemy(obs, position[1])))
                    distances = sorted(distances, key=lambda tup: tup[1])
                    for distance in distances:
                        if distance[1] > enemy_range:
                            self.set_target_destination(self.movement_step(
                                distance[0], enemy_range + movement_step, obs))
                            return move["function"]
                else:
                    return move["function"]

        if self.can_do_action(obs, actions.FUNCTIONS.select_army.id):
            return actions.FUNCTIONS.select_army("select")

    def calculate_fitness(self, obs):
        """
        fitness = ((total damage dealt - hit point loss)/ initial hit points) + 1
        :param obs: observation from sc2 env
        :return: current fitness score in range [0,2]
        """
        feature_units = obs.observation.feature_units
        total_damage_dealt = self.initial_enemy_hit_points - \
                             self.calculate_hitpoints(feature_units, _PLAYER_ENEMY)
        hit_point_loss = self.initial_self_hit_points - \
                         self.calculate_hitpoints(feature_units, _PLAYER_SELF)
        self.fitness = (total_damage_dealt - hit_point_loss) / \
                       self.initial_self_hit_points + 1
        return self.fitness


class CombatAgent2(BaseNeatAgent):
    def __init__(self, **kwargs):
        super(CombatAgent2, self).__init__(**kwargs)
        self.id = kwargs.get('id', 0)
        self.initial_self_hit_points = 0
        self.initial_enemy_hit_points = 0

    def fitness_calculation_setup(self, obs):
        feature_units = obs.observation.feature_units
        self.initial_self_hit_points = self.calculate_hitpoints(
            feature_units, _PLAYER_SELF)
        self.initial_enemy_hit_points = self.calculate_hitpoints(
            feature_units, _PLAYER_ENEMY)

    @staticmethod
    def retrieve_enemy_location(obs):
        enemy_y, enemy_x = (
                obs.observation.feature_screen.player_relative == _PLAYER_ENEMY).nonzero()
        enemy = list(zip(enemy_x, enemy_y))
        enemy_center = np.mean(enemy, axis=0).round()

        return enemy_center

    @staticmethod
    def retrieve_distance_to_enemy(obs, pos):
        enemy_center = CombatAgent2.retrieve_enemy_location(obs)

        distances = np.linalg.norm(np.array(enemy_center) - pos)

        return distances.min()

    def retrieve_handcrafted_inputs(self, obs):
        """:returns
        [
        current_hp,
         weapon_cooldown,
         enemy_in_range
         ]
        """
        feature_units = obs.observation.feature_units
        allies = [unit for unit in feature_units if unit.alliance == _PLAYER_SELF]

        current_hp = self.calculate_hitpoints(feature_units, _PLAYER_SELF)
        current_hp = current_hp / self.initial_self_hit_points

        weapon_cooldown = 0
        for ally in allies:
            if ally.weapon_cooldown > 0:
                weapon_cooldown += 1

        if weapon_cooldown > (len(allies) / 2):
            # nn input weapon cooldown = 1 means the majority cannot fire
            weapon_cooldown = 1
        else:
            weapon_cooldown = 0

        weapon_range = 5
        radius = 2.5

        if self.retrieve_distance_to_enemy(obs, self.get_current_location(obs)) < (weapon_range + 2 * radius):
            enemy_in_range = 1
        else:
            enemy_in_range = 0

        return [current_hp, weapon_cooldown, enemy_in_range]

    def step(self, obs):
        """
        network controls fight or flee decision with first output and then displacement magnitude and direction with
        the other two outputs
        """
        self.obs = obs
        decision = self.nn_output[0]
        # sigmoid activation function so need to subtract 0.5 to allow movement in all directions
        displacement = [self.nn_output[1] - 0.5, self.nn_output[2] - 0.5]

        if self.first_move:
            if self.can_do_action(obs, actions.FUNCTIONS.Move_screen.id):
                self.first_move = False
                self.set_target_destination(self.retrieve_enemy_location(obs))
                return self.move_unit(obs)["function"]

        if decision > 0.5:  # fight
            if self.can_do_action(obs, actions.FUNCTIONS.Attack_screen.id):
                player_relative = obs.observation.feature_screen.player_relative
                enemy = self.xy_locs(player_relative == _PLAYER_ENEMY)
                if not enemy:
                    return actions.FUNCTIONS.no_op()

                target = enemy[np.argmax(np.array(enemy)[:, 1])]
                return actions.FUNCTIONS.Attack_screen("now", target)
        else:  # flee
            if self.can_do_action(obs, actions.FUNCTIONS.Move_screen.id):
                move = self.move_unit(obs)
                if move["status"] is "ARRIVED_AT_TARGET":
                    step_size = 10
                    self.movement_step(step_size, displacement, obs)
                    return actions.FUNCTIONS.Move_screen("now", (self.target[0], self.target[1]))
                else:
                    return move["function"]

        if self.can_do_action(obs, actions.FUNCTIONS.select_army.id):
            return actions.FUNCTIONS.select_army("select")

    def calculate_fitness(self, obs):
        """
        fitness = ((total damage dealt - hit point loss)/ initial hit points) + 1
        :param obs: observation from sc2 env
        :return: current fitness score in range [0,2]
        """
        feature_units = obs.observation.feature_units
        self.fitness = self.initial_enemy_hit_points + self.calculate_hitpoints(
            feature_units, _PLAYER_SELF) - self.calculate_hitpoints(feature_units, _PLAYER_ENEMY)
        return self.fitness


class CombatAgent3(BaseNeatAgent):
    def __init__(self, **kwargs):
        super(CombatAgent3, self).__init__(**kwargs)
        self.id = kwargs.get('id', 0)
        self.initial_self_hit_points = 0
        self.initial_enemy_hit_points = 0
        self.previous_command = "FIGHT"

    def fitness_calculation_setup(self, obs):
        feature_units = obs.observation.feature_units
        self.initial_self_hit_points = self.calculate_hitpoints(
            feature_units, _PLAYER_SELF)
        self.initial_enemy_hit_points = self.calculate_hitpoints(
            feature_units, _PLAYER_ENEMY)

    @staticmethod
    def retrieve_enemy_location(obs):
        enemy_y, enemy_x = (
                obs.observation.feature_screen.player_relative == _PLAYER_ENEMY).nonzero()
        enemy = list(zip(enemy_x, enemy_y))
        enemy_center = np.mean(enemy, axis=0).round()

        return enemy_center

    @staticmethod
    def retrieve_distance_to_enemy(obs, pos):
        enemy_center = CombatAgent3.retrieve_enemy_location(obs)

        distances = np.linalg.norm(np.array(enemy_center) - pos)

        return distances.min()

    def calculate_distance_to_bounds(self, obs):
        """
        calculates the distance to four map boundaries from players average position
        y distance for N,S
        x distance for W,E
        :param obs: sc2 observation
        :return: 4 values corresponding to boundary distance in order N,S,W,E. always positive
        """
        position = self.get_current_location(obs)

        # top-left of map is (0,0) and bottom-right is (map_width-1, map_height-1)
        def calc_bound_dist(bound_direction):
            return {
                "NORTH": lambda x: position[1],
                "SOUTH": lambda x: x.max_map_height - position[1],
                "WEST": lambda x: position[0],
                "EAST": lambda x: x.max_map_width - position[0]
            }[bound_direction](self)

        distances = {}
        bounds = ["NORTH", "SOUTH", "WEST", "EAST"]
        for bound in bounds:
            distances[bound] = calc_bound_dist(bound) / self.max_map_height

        return distances["NORTH"], distances["SOUTH"], distances["WEST"], distances["EAST"]

    def retrieve_handcrafted_inputs(self, obs):
        """:returns
        [
         current_hp,
         weapon_cooldown,
         enemy_in_range,
         previous_command,
         north_bound,
         south_bound,
         west_bound,
         east_bound
        ]
        """
        feature_units = obs.observation.feature_units
        allies = [unit for unit in feature_units if unit.alliance == _PLAYER_SELF]

        current_hp = self.calculate_hitpoints(feature_units, _PLAYER_SELF)
        current_hp = current_hp / self.initial_self_hit_points

        weapon_cooldown = 0
        for ally in allies:
            if ally.weapon_cooldown > 0:
                weapon_cooldown += 1

        if weapon_cooldown > (len(allies) / 2):
            # nn input weapon cooldown = 1 means the majority cannot fire
            weapon_cooldown = 1
        else:
            weapon_cooldown = 0

        weapon_range = 5
        radius = 2.5

        if self.retrieve_distance_to_enemy(obs, self.get_current_location(obs)) < (weapon_range + 2 * radius):
            enemy_in_range = 1
        else:
            enemy_in_range = 0

        north_bound, south_bound, west_bound, east_bound = self.calculate_distance_to_bounds(obs)

        if self.previous_command == "FIGHT":
            prev_cmd = 1
        elif self.previous_command == "FLEE":
            prev_cmd = 0

        return [current_hp, weapon_cooldown, enemy_in_range, prev_cmd, north_bound, south_bound, west_bound, east_bound]

    def step(self, obs):
        self.obs = obs
        decision = self.nn_output[0]

        # sigmoid activation function so need to subtract 0.5 to allow movement in all directions
        # (ALSO CHANGE BOOLEAN DECISION CHECK to 0.5)
        # displacement = [self.nn_output[1] - 0.5, self.nn_output[2] - 0.5]

        # clamped or tanh (ALSO CHANGE BOOLEAN DECISION CHECK to 0)
        displacement = [self.nn_output[1], self.nn_output[2]]

        if self.first_move:
            if self.can_do_action(obs, actions.FUNCTIONS.Move_screen.id):
                self.first_move = False
                self.set_target_destination(self.retrieve_enemy_location(obs))
                return self.move_unit(obs)["function"]

        if decision > 0:  # fight
            self.previous_command = "FIGHT"
            if self.can_do_action(obs, actions.FUNCTIONS.Attack_screen.id):
                player_relative = obs.observation.feature_screen.player_relative
                enemy = self.xy_locs(player_relative == _PLAYER_ENEMY)
                if not enemy:
                    return actions.FUNCTIONS.no_op()

                target = enemy[np.argmax(np.array(enemy)[:, 1])]
                return actions.FUNCTIONS.Attack_screen("now", target)
        else:  # flee
            self.previous_command = "FLEE"
            if self.can_do_action(obs, actions.FUNCTIONS.Move_screen.id):
                move = self.move_unit(obs)
                if move["status"] is "ARRIVED_AT_TARGET":
                    step_size = 10
                    self.movement_step(step_size, displacement, obs)
                    return actions.FUNCTIONS.Move_screen("now", (self.target[0], self.target[1]))
                else:
                    return move["function"]

        if self.can_do_action(obs, actions.FUNCTIONS.select_army.id):
            return actions.FUNCTIONS.select_army("select")

    def calculate_fitness(self, obs):
        """
        fitness = init_enemy_hitpoints + current_self_hitpoints - current_enemy_hitpoints
        :param obs: observation from sc2 env
        :return: current fitness score (scaled hardcoding)
        """
        feature_units = obs.observation.feature_units
        self.fitness = self.initial_enemy_hit_points + self.calculate_hitpoints(
            feature_units, _PLAYER_SELF) - self.calculate_hitpoints(feature_units, _PLAYER_ENEMY)
        return self.fitness / 950


class CombatAgent4(BaseNeatAgent):
    """Version that successfully learnt kiting for ranged vs melee"""

    def __init__(self, **kwargs):
        super(CombatAgent4, self).__init__(**kwargs)
        self.id = kwargs.get('id', 0)
        self.initial_self_hit_points = 0
        self.initial_enemy_hit_points = 0
        self.previous_command = "FIGHT"
        self.forced_engage = True

    def fitness_calculation_setup(self, obs):
        feature_units = obs.observation.feature_units
        self.initial_self_hit_points = self.calculate_hitpoints(
            feature_units, _PLAYER_SELF)
        self.initial_enemy_hit_points = self.calculate_hitpoints(
            feature_units, _PLAYER_ENEMY)

        self.max_fitness = self.initial_self_hit_points + self.initial_enemy_hit_points

        self.genome_threshold = (self.initial_enemy_hit_points + 1) / float(self.max_fitness)

    @staticmethod
    def retrieve_enemy_location(obs):
        enemy_y, enemy_x = (
                obs.observation.feature_screen.player_relative == _PLAYER_ENEMY).nonzero()
        enemy = list(zip(enemy_x, enemy_y))
        enemy_center = np.mean(enemy, axis=0).round()

        return enemy_center

    @staticmethod
    def retrieve_distance_to_enemy(obs, pos):
        enemy_center = CombatAgent4.retrieve_enemy_location(obs)

        distances = np.linalg.norm(np.array(enemy_center) - pos)

        return distances.min()

    def calculate_distance_to_bounds(self, obs):
        """
        calculates the distance to four map boundaries from players average position
        y distance for N,S
        x distance for W,E
        :param obs: sc2 observation
        :return: 4 values corresponding to boundary distance in order N,S,W,E. always positive
        """
        position = self.get_current_location(obs)

        # top-left of map is (0,0) and bottom-right is (map_width-1, map_height-1)
        def calc_bound_dist(bound_direction):
            return {
                "NORTH": lambda x: position[1],
                "SOUTH": lambda x: x.max_map_height - position[1],
                "WEST": lambda x: position[0],
                "EAST": lambda x: x.max_map_width - position[0]
            }[bound_direction](self)

        distances = {}
        bounds = ["NORTH", "SOUTH", "WEST", "EAST"]
        for bound in bounds:
            if bound == "NORTH" or bound == "SOUTH":
                distances[bound] = calc_bound_dist(bound) / self.max_map_height
            elif bound == "EAST" or bound == "WEST":
                distances[bound] = calc_bound_dist(bound) / self.max_map_width

        return distances["NORTH"], distances["SOUTH"], distances["WEST"], distances["EAST"]

    def detect_enemies_by_region(self, obs):
        """
        checks if enemies present in the four regions around players average position
        :param obs: sc2 observation
        :return: 4 values of enemy detection booleans inorder of NW, NE, SW, SE.
        """
        position = self.get_current_location(obs)
        enemy_y, enemy_x = (
                obs.observation.feature_screen.player_relative == _PLAYER_ENEMY).nonzero()
        enemies = list(zip(enemy_x, enemy_y))

        # top-left of map is (0,0) and bottom-right is (map_width-1, map_height-1)
        def detect_enemies(region_direction):
            for enemy in enemies:
                if region_direction == "NORTH_WEST":
                    if enemy[0] < position[0] and enemy[1] < position[1]:
                        return 1
                elif region_direction == "NORTH_EAST":
                    if enemy[0] >= position[0] and enemy[1] < position[1]:
                        return 1
                elif region_direction == "SOUTH_WEST":
                    if enemy[0] < position[0] and enemy[1] >= position[1]:
                        return 1
                elif region_direction == "SOUTH_EAST":
                    if enemy[0] >= position[0] and enemy[1] >= position[1]:
                        return 1
            return 0

        presences = {}
        regions = ["NORTH_WEST", "NORTH_EAST", "SOUTH_WEST", "SOUTH_EAST"]
        for region in regions:
            presences[region] = detect_enemies(region)

        return presences["NORTH_WEST"], presences["NORTH_EAST"], presences["SOUTH_WEST"], presences["SOUTH_EAST"]

    def retrieve_handcrafted_inputs(self, obs):
        """:returns
        [
         current_hp - cumulative,
         weapon_cooldown - consider majority,
         enemy_in_range,
         previous_command,
         north_bound,
         south_bound,
         west_bound,
         east_bound,
         nw_enemy_presence,
         ne_enemy_presence,
         sw_enemy_presence,
         se_enemy_presence
        ] all scaled [0,1]
        """
        feature_units = obs.observation.feature_units
        allies = [unit for unit in feature_units if unit.alliance == _PLAYER_SELF]
        enemies = [unit for unit in feature_units if unit.alliance == _PLAYER_ENEMY]

        current_hp = self.calculate_hitpoints(feature_units, _PLAYER_SELF)
        current_hp = current_hp / self.initial_self_hit_points

        weapon_cooldown = 0
        for ally in allies:
            if ally.weapon_cooldown > 0:
                weapon_cooldown += 1
        if weapon_cooldown > (len(allies) / 2):
            # nn input weapon cooldown = 1 means the majority cannot fire
            weapon_cooldown = 1
        else:
            weapon_cooldown = 0

        # assumes our units are all the same type
        weapon_range = 5
        self_radius = 1
        if len(allies) > 0:
            weapon_range = weapon_ranges[allies[0].unit_type]
            self_radius = unit_sizes[allies[0].unit_type] / float(2)

        enemy_radius = 1
        if len(enemies) > 0:
            enemy_radius = unit_sizes[enemies[0].unit_type] / float(2)

        if self.retrieve_distance_to_enemy(obs, self.get_current_location(obs)) < (
                self_radius + weapon_range + enemy_radius):
            enemy_in_range = 1
        else:
            enemy_in_range = 0

        north_bound, south_bound, west_bound, east_bound = self.calculate_distance_to_bounds(obs)

        if self.previous_command == "FIGHT":
            prev_cmd = 1
        elif self.previous_command == "FLEE":
            prev_cmd = 0

        nw_enemy_presence, ne_enemy_presence, sw_enemy_presence, se_enemy_presence = self.detect_enemies_by_region(obs)

        return [current_hp, weapon_cooldown, enemy_in_range, prev_cmd, north_bound, south_bound, west_bound, east_bound,
                nw_enemy_presence, ne_enemy_presence, sw_enemy_presence, se_enemy_presence]

    def step(self, obs):
        self.obs = obs

        if self.train and np.random.rand() < 0.02:  # random exploration
            decision = np.random.randint(low=-1, high=1, size=1)
            displacement = [np.random.randint(low=-1, high=1, size=1), np.random.randint(low=-1, high=1, size=1)]
        else:
            decision = self.nn_output[0]

            # sigmoid activation function so need to subtract 0.5 to allow movement in all directions
            # (ALSO CHANGE BOOLEAN DECISION CHECK to 0.5)
            # displacement = [self.nn_output[1] - 0.5, self.nn_output[2] - 0.5]

            # clamped or tanh (ALSO CHANGE BOOLEAN DECISION CHECK to 0)
            displacement = [self.nn_output[1], self.nn_output[2]]
            # to have constant magnitude uncomment the following for loop
            # for delta in displacement:
            #     if delta > 0:
            #         delta = 0.5
            #     else:
            #         delta = -0.5

        # to approach enemy on the first move of the episode
        if self.first_move:
            if self.forced_engage:
                if self.can_do_action(obs, actions.FUNCTIONS.Move_screen.id):
                    self.first_move = False
                    enemy_location = self.retrieve_enemy_location(obs)
                    current_location = self.get_current_location(obs)
                    if current_location[0] < enemy_location[0]:
                        target_location = [enemy_location[0] - 6, enemy_location[1]]
                    else:
                        target_location = [enemy_location[0] + 6, enemy_location[1]]

                    self.set_target_destination(target_location)
                    return self.move_unit(obs)["function"]

        if decision > 0:  # fight
            self.previous_command = "FIGHT"
            if self.can_do_action(obs, actions.FUNCTIONS.Attack_screen.id):
                features = obs.observation.feature_units
                enemies = [unit for unit in features if unit.alliance == _PLAYER_ENEMY]
                if not enemies:
                    return actions.FUNCTIONS.no_op()
                enemies = sorted(enemies, key=lambda x: x.health)
                lowest_health_enemy_coords = [enemies[0].x, enemies[0].y]
                return actions.FUNCTIONS.Attack_screen("now", lowest_health_enemy_coords)
        else:  # flee
            self.previous_command = "FLEE"
            if self.can_do_action(obs, actions.FUNCTIONS.Move_screen.id):
                move = self.move_unit(obs)
                if move["status"] is "ARRIVED_AT_TARGET":
                    step_size = 10
                    self.movement_step(step_size, displacement, obs)
                    return actions.FUNCTIONS.Move_screen("now", (self.target[0], self.target[1]))
                else:
                    return move["function"]

        if self.can_do_action(obs, actions.FUNCTIONS.select_army.id):
            return actions.FUNCTIONS.select_army("select")

    def calculate_fitness(self, obs):
        """
        fitness = init_enemy_hitpoints + current_self_hitpoints - current_enemy_hitpoints
        :param obs: observation from sc2 env
        :return: current fitness score
        """
        feature_units = obs.observation.feature_units
        self.fitness = self.initial_enemy_hit_points + self.calculate_hitpoints(
            feature_units, _PLAYER_SELF) - self.calculate_hitpoints(feature_units, _PLAYER_ENEMY)
        max_fitness = self.max_fitness
        return self.fitness / max_fitness


class RefinedCombatAgent1(CombatAgent4):
    def __init__(self, **kwargs):
        super(RefinedCombatAgent1, self).__init__(**kwargs)

    def retrieve_handcrafted_inputs(self, obs):
        """
        assumes units are homogenous
        :param obs:
        :returns
        [
         current_hp - cumulative, s
         weapon_cooldown - consider majority, s
         enemy_in_range - consider average ranges, s
         previous_command, s
         north_bound, s
         south_bound, s
         west_bound, s
         east_bound, s
         nw_enemy_presence, s
         ne_enemy_presence, s
         sw_enemy_presence, s
         se_enemy_presence, s
         self_unit_type - 1 for ranged, s
         enemy_unit_type - 1 for ranged, s
         self_weapon_range,
         enemy_weapon_range,
         self_speed,
         enemy_speed
        ] s for scaled [0,1]
        """
        feature_units = obs.observation.feature_units
        allies = [unit for unit in feature_units if unit.alliance == _PLAYER_SELF]
        enemies = [unit for unit in feature_units if unit.alliance == _PLAYER_ENEMY]

        current_hp = self.calculate_hitpoints(feature_units, _PLAYER_SELF)
        current_hp = current_hp / self.initial_self_hit_points

        weapon_cooldown = 0
        for ally in allies:
            if ally.weapon_cooldown > 0:
                weapon_cooldown += 1
        if weapon_cooldown > (len(allies) / 2):
            # nn input weapon cooldown = 1 means the majority cannot fire
            weapon_cooldown = 1
        else:
            weapon_cooldown = 0

        self_weapon_range = 5
        self_radius = 1
        self_unit_type = 1
        self_speed = 1
        if len(allies) > 0:
            self_weapon_range = weapon_ranges[allies[0].unit_type]
            self_radius = unit_sizes[allies[0].unit_type] / float(2)
            self_unit_type = unit_type[allies[0].unit_type]
            self_speed = unit_speed[allies[0].unit_type]

        enemy_radius = 1
        enemy_weapon_range = 1
        enemy_unit_type = 0
        enemy_speed = 1
        if len(enemies) > 0:
            enemy_weapon_range = weapon_ranges[enemies[0].unit_type]
            enemy_radius = unit_sizes[enemies[0].unit_type] / float(2)
            enemy_unit_type = unit_type[enemies[0].unit_type]
            enemy_speed = unit_speed[enemies[0].unit_type]

        if self.retrieve_distance_between_positions(self.retrieve_enemy_location(obs),
                                                    self.get_current_location(obs)) < (
                self_radius + self_weapon_range + enemy_radius):
            enemy_in_range = 1
        else:
            enemy_in_range = 0

        north_bound, south_bound, west_bound, east_bound = self.calculate_distance_to_bounds(obs)

        if self.previous_command == "FIGHT":
            prev_cmd = 1
        elif self.previous_command == "FLEE":
            prev_cmd = 0

        nw_enemy_presence, ne_enemy_presence, sw_enemy_presence, se_enemy_presence = self.detect_enemies_by_region(obs)

        return [current_hp, weapon_cooldown, enemy_in_range, prev_cmd, north_bound, south_bound, west_bound, east_bound,
                nw_enemy_presence, ne_enemy_presence, sw_enemy_presence, se_enemy_presence, self_unit_type,
                enemy_unit_type, self_weapon_range, enemy_weapon_range, self_speed, enemy_speed]


class RefinedCombatAgent2(CombatAgent4):
    """difference from refined agent 1 is that it doesnt have unit speed inputs and has an in enemy range input"""

    def __init__(self, **kwargs):
        super(RefinedCombatAgent2, self).__init__(**kwargs)

    def retrieve_enemy_location(self, obs):
        player_relative = obs.observation.feature_screen.player_relative
        player_y, player_x = (
                player_relative == _PLAYER_ENEMY).nonzero()
        pos = [int(player_x.mean()), int(player_y.mean())]
        return pos

    def retrieve_handcrafted_inputs(self, obs):
        """
        assumes units are homogenous
        :param obs:
        :returns
        [
         current_hp - cumulative, s
         weapon_cooldown - consider majority, s
         enemy_in_range - consider average ranges, s
         in_enemy_range - consider average ranges, s
         previous_command, s
         north_bound, s
         south_bound, s
         west_bound, s
         east_bound, s
         nw_enemy_presence, s
         ne_enemy_presence, s
         sw_enemy_presence, s
         se_enemy_presence, s
         self_unit_type - 1 for ranged, s
         enemy_unit_type - 1 for ranged, s
        ] s for scaled [0,1]
        """
        feature_units = obs.observation.feature_units
        allies = [unit for unit in feature_units if unit.alliance == _PLAYER_SELF]
        enemies = [unit for unit in feature_units if unit.alliance == _PLAYER_ENEMY]

        current_hp = self.calculate_hitpoints(feature_units, _PLAYER_SELF)
        current_hp = current_hp / self.initial_self_hit_points

        weapon_cooldown = 0
        for ally in allies:
            if ally.weapon_cooldown > 0:
                weapon_cooldown += 1
        if weapon_cooldown > (len(allies) / 2):
            # nn input weapon cooldown = 1 means the majority cannot fire
            weapon_cooldown = 1
        else:
            weapon_cooldown = 0

        self_weapon_range = 5
        self_radius = 1
        self_unit_type = 1
        if len(allies) > 0:
            self_weapon_range = weapon_ranges[allies[0].unit_type]
            self_radius = unit_sizes[allies[0].unit_type] / float(2)
            self_unit_type = unit_type[allies[0].unit_type]

        enemy_radius = 1
        enemy_weapon_range = 1
        enemy_unit_type = 0
        if len(enemies) > 0:
            enemy_weapon_range = weapon_ranges[enemies[0].unit_type]
            enemy_radius = unit_sizes[enemies[0].unit_type] / float(2)
            enemy_unit_type = unit_type[enemies[0].unit_type]

        if self.retrieve_distance_between_positions(self.retrieve_enemy_location(obs),
                                                    self.get_current_location(obs)) < (
                self_radius + self_weapon_range + enemy_radius):
            enemy_in_range = 1
        else:
            enemy_in_range = 0

        if self.retrieve_distance_between_positions(self.retrieve_enemy_location(obs),
                                                    self.get_current_location(obs)) < (
                self_radius + enemy_weapon_range + enemy_radius):
            in_enemy_range = 1
        else:
            in_enemy_range = 0

        north_bound, south_bound, west_bound, east_bound = self.calculate_distance_to_bounds(obs)

        if self.previous_command == "FIGHT":
            prev_cmd = 1
        elif self.previous_command == "FLEE":
            prev_cmd = 0

        nw_enemy_presence, ne_enemy_presence, sw_enemy_presence, se_enemy_presence = self.detect_enemies_by_region(obs)

        return [current_hp, weapon_cooldown, enemy_in_range, in_enemy_range, prev_cmd, north_bound, south_bound,
                west_bound, east_bound,
                nw_enemy_presence, ne_enemy_presence, sw_enemy_presence, se_enemy_presence, self_unit_type,
                enemy_unit_type]


class RefinedCombatAgent3(CombatAgent4):
    """difference from refined agent 1 is that it also has unit type id"""

    def __init__(self, **kwargs):
        super(RefinedCombatAgent3, self).__init__(**kwargs)
        self.self_id = 0
        self.enemy_id = 0

    # def retrieve_enemy_location(self, obs):
    #     player_relative = obs.observation.feature_screen.player_relative
    #     player_y, player_x = (
    #             player_relative == _PLAYER_ENEMY).nonzero()
    #     pos = [int(player_x.mean()), int(player_y.mean())]
    #     return pos

    @staticmethod
    def retrieve_enemy_location(obs):
        enemy_y, enemy_x = (
                obs.observation.feature_screen.player_relative == _PLAYER_ENEMY).nonzero()
        enemy = list(zip(enemy_x, enemy_y))
        enemy_center = np.mean(enemy, axis=0).round()

        return enemy_center

    def retrieve_handcrafted_inputs(self, obs):
        """
        assumes units are homogenous
        :param obs:
        :returns
        [
         current_hp - cumulative, s
         weapon_cooldown - consider majority, s
         enemy_in_range - consider average ranges, s
         in_enemy_range - consider average ranges, s ==========
         previous_command, s
         north_bound, s
         south_bound, s
         west_bound, s
         east_bound, s
         nw_enemy_presence, s
         ne_enemy_presence, s
         sw_enemy_presence, s
         se_enemy_presence, s
         self_unit_type - 1 for ranged, s =====
         enemy_unit_type - 1 for ranged, s ===
         self_weapon_range, ====
         enemy_weapon_range, ====
         self_speed, ===
         enemy_speed, ===
         self_id, ===
         enemy_id ===
        ] s for scaled [0,1]
        """
        feature_units = obs.observation.feature_units
        allies = [unit for unit in feature_units if unit.alliance == _PLAYER_SELF]
        enemies = [unit for unit in feature_units if unit.alliance == _PLAYER_ENEMY]

        current_hp = self.calculate_hitpoints(feature_units, _PLAYER_SELF)
        current_hp = current_hp / self.initial_self_hit_points

        weapon_cooldown = 0
        for ally in allies:
            if ally.weapon_cooldown > 0:
                weapon_cooldown += 1
        if weapon_cooldown > (len(allies) / 2):
            # nn input weapon cooldown = 1 means the majority cannot fire
            weapon_cooldown = 1
        else:
            weapon_cooldown = 0

        self_weapon_range = 5
        self_radius = 1
        self_unit_type = 1
        self_speed = 1
        if len(allies) > 0:
            self.self_id = allies[0].unit_type
            self_weapon_range = weapon_ranges[self.self_id]
            self_radius = unit_sizes[self.self_id] / float(2)
            self_unit_type = unit_type[self.self_id]
            self_speed = unit_speed[self.self_id]

        enemy_radius = 1
        enemy_weapon_range = 1
        enemy_unit_type = 0
        enemy_speed = 1
        if len(enemies) > 0:
            self.enemy_id = enemies[0].unit_type
            enemy_weapon_range = weapon_ranges[self.enemy_id]
            enemy_radius = unit_sizes[self.enemy_id] / float(2)
            enemy_unit_type = unit_type[self.enemy_id]
            enemy_speed = unit_speed[self.enemy_id]

        if self.retrieve_distance_between_positions(self.retrieve_enemy_location(obs),
                                                    self.get_current_location(obs)) < (
                self_radius + self_weapon_range + enemy_radius):
            enemy_in_range = 1
        else:
            enemy_in_range = 0

        in_enemy_range = 0
        for ally in allies:
            for enemy in enemies:
                if self.retrieve_distance_between_positions([enemy.x, enemy.y], [ally.x, ally.y]) < (
                        self_radius + enemy_weapon_range + enemy_radius):
                    in_enemy_range = 1
                    break
                else:
                    in_enemy_range = 0

        north_bound, south_bound, west_bound, east_bound = self.calculate_distance_to_bounds(obs)

        if self.previous_command == "FIGHT":
            prev_cmd = 1
        elif self.previous_command == "FLEE":
            prev_cmd = 0

        nw_enemy_presence, ne_enemy_presence, sw_enemy_presence, se_enemy_presence = self.detect_enemies_by_region(obs)

        return [current_hp, weapon_cooldown, enemy_in_range, in_enemy_range, prev_cmd, north_bound, south_bound,
                west_bound, east_bound,
                nw_enemy_presence, ne_enemy_presence, sw_enemy_presence, se_enemy_presence, self_unit_type,
                enemy_unit_type, self_weapon_range, enemy_weapon_range, self_speed, enemy_speed, self.self_id,
                self.enemy_id]


class HeteroCombatAgent(CombatAgent4):
    """can accommodate hetero compositions of units for self by cycling our units every other step"""

    def __init__(self, **kwargs):
        super(HeteroCombatAgent, self).__init__(**kwargs)
        self.current_group_id = 0
        self.unit_type_ids = []
        self.is_select_step = True
        self.unit_group_select_counter = 0
        self.enemy_id = 0
        self.previous_commands = {}
        self.init_unit_counts = {}

    def fitness_calculation_setup(self, obs):
        feature_units = obs.observation.feature_units
        self.initial_self_hit_points = self.calculate_hitpoints(
            feature_units, _PLAYER_SELF)
        self.initial_enemy_hit_points = self.calculate_hitpoints(
            feature_units, _PLAYER_ENEMY)

        self.max_fitness = self.initial_self_hit_points + self.initial_enemy_hit_points

        self.genome_threshold = (self.initial_enemy_hit_points + 1) / float(self.max_fitness)

        for unit in obs.observation.feature_units:
            if unit.alliance == _PLAYER_SELF:
                self.previous_commands[unit.unit_type] = "FIGHT"

    # def retrieve_enemy_location(self, obs):
    #     player_relative = obs.observation.feature_screen.player_relative
    #     player_y, player_x = (
    #             player_relative == _PLAYER_ENEMY).nonzero()
    #     pos = [int(player_x.mean()), int(player_y.mean())]
    #     return pos

    @staticmethod
    def retrieve_enemy_location(obs):
        enemy_y, enemy_x = (
                obs.observation.feature_screen.player_relative == _PLAYER_ENEMY).nonzero()
        enemy = list(zip(enemy_x, enemy_y))
        enemy_center = np.mean(enemy, axis=0).round()

        return enemy_center

    def calculate_distance_to_bounds(self, obs, for_subgroup=False):
        """
        calculates the distance to four map boundaries from players average position
        y distance for N,S
        x distance for W,E
        :param for_subgroup: if being used for army or currently selected subgroup
        :param obs: sc2 observation
        :return: 4 values corresponding to boundary distance in order N,S,W,E. always positive
        """
        if for_subgroup:
            position = self.get_avg_location_of_self_subgroup(obs)
        else:
            position = self.get_current_location(obs)

        # top-left of map is (0,0) and bottom-right is (map_width-1, map_height-1)
        def calc_bound_dist(bound_direction):
            return {
                "NORTH": lambda x: position[1],
                "SOUTH": lambda x: x.max_map_height - position[1],
                "WEST": lambda x: position[0],
                "EAST": lambda x: x.max_map_width - position[0]
            }[bound_direction](self)

        distances = {}
        bounds = ["NORTH", "SOUTH", "WEST", "EAST"]
        for bound in bounds:
            if bound == "NORTH" or bound == "SOUTH":
                distances[bound] = calc_bound_dist(bound) / self.max_map_height
            elif bound == "EAST" or bound == "WEST":
                distances[bound] = calc_bound_dist(bound) / self.max_map_width

        return distances["NORTH"], distances["SOUTH"], distances["WEST"], distances["EAST"]

    def detect_enemies_by_region(self, obs, for_subgroup=False):
        """
        checks if enemies present in the four regions around players average position
        :param for_subgroup: if being used for army or currently selected subgroup
        :param obs: sc2 observation
        :return: 4 values of enemy detection booleans inorder of NW, NE, SW, SE.
        """
        if for_subgroup:
            position = self.get_avg_location_of_self_subgroup(obs)
        else:
            position = self.get_current_location(obs)
        enemy_y, enemy_x = (
                obs.observation.feature_screen.player_relative == _PLAYER_ENEMY).nonzero()
        enemies = list(zip(enemy_x, enemy_y))

        # top-left of map is (0,0) and bottom-right is (map_width-1, map_height-1)
        def detect_enemies(region_direction):
            for enemy in enemies:
                if region_direction == "NORTH_WEST":
                    if enemy[0] < position[0] and enemy[1] < position[1]:
                        return 1
                elif region_direction == "NORTH_EAST":
                    if enemy[0] >= position[0] and enemy[1] < position[1]:
                        return 1
                elif region_direction == "SOUTH_WEST":
                    if enemy[0] < position[0] and enemy[1] >= position[1]:
                        return 1
                elif region_direction == "SOUTH_EAST":
                    if enemy[0] >= position[0] and enemy[1] >= position[1]:
                        return 1
            return 0

        presences = {}
        regions = ["NORTH_WEST", "NORTH_EAST", "SOUTH_WEST", "SOUTH_EAST"]
        for region in regions:
            presences[region] = detect_enemies(region)

        return presences["NORTH_WEST"], presences["NORTH_EAST"], presences["SOUTH_WEST"], presences["SOUTH_EAST"]

    def retrieve_handcrafted_inputs(self, obs):
        """
        retrieves info with respect with currently selected subgroup
        :param obs:
        :returns
        [
         current_hp - cumulative, s
         weapon_cooldown - consider majority, s
         enemy_in_range - consider average ranges, s
         in_enemy_range - consider average ranges, s
         previous_command, s
         north_bound, s
         south_bound, s
         west_bound, s
         east_bound, s
         nw_enemy_presence, s
         ne_enemy_presence, s
         sw_enemy_presence, s
         se_enemy_presence, s
         self_unit_type - 1 for ranged, s
         enemy_unit_type - 1 for ranged, s
         self_weapon_range,
         enemy_weapon_range,
         self_speed,
         enemy_speed,
         distance_to_enemy
        ] s for scaled [0,1]
        """
        self.detect_self_unit_types(obs)

        feature_units = obs.observation.feature_units
        allies = [unit for unit in feature_units if unit.alliance == _PLAYER_SELF]
        selected_allies = [unit for unit in allies if unit.unit_type == self.current_group_id]
        enemies = [unit for unit in feature_units if unit.alliance == _PLAYER_ENEMY]

        hitpoints = 0
        for unit in selected_allies:
            hitpoints += unit.health

        if self.current_group_id in unit_health.keys():
            init_hp = 0
            init_hp = unit_health[self.current_group_id] * self.init_unit_counts[self.current_group_id]
        else:
            init_hp = self.initial_self_hit_points
        current_hp = hitpoints / init_hp

        weapon_cooldown = 0
        for ally in selected_allies:
            if ally.weapon_cooldown > 0:
                weapon_cooldown += 1
        if weapon_cooldown > (len(selected_allies) / 2):
            # nn input weapon cooldown = 1 means the majority cannot fire
            weapon_cooldown = 1
        else:
            weapon_cooldown = 0

        self_weapon_range = 5
        self_radius = 1
        self_unit_type = 1
        self_speed = 1
        if len(selected_allies) > 0:
            self_weapon_range = weapon_ranges[self.current_group_id]
            self_radius = unit_sizes[self.current_group_id] / float(2)
            self_unit_type = unit_type[self.current_group_id]
            self_speed = unit_speed[self.current_group_id]

        enemy_radius = 1
        enemy_weapon_range = 1
        enemy_unit_type = 0
        enemy_speed = 1
        if len(enemies) > 0:
            self.enemy_id = enemies[0].unit_type
            enemy_weapon_range = weapon_ranges[self.enemy_id]
            enemy_radius = unit_sizes[self.enemy_id] / float(2)
            enemy_unit_type = unit_type[self.enemy_id]
            enemy_speed = unit_speed[self.enemy_id]

        # TODO can be inaccurate if using melee units
        if self.retrieve_distance_between_positions(self.retrieve_enemy_location(obs),
                                                    self.get_avg_location_of_self_subgroup(obs)) < (
                self_radius + self_weapon_range + enemy_radius):
            enemy_in_range = 1
        else:
            enemy_in_range = 0

        in_enemy_range = 0
        for ally in selected_allies:
            for enemy in enemies:
                if self.retrieve_distance_between_positions([enemy.x, enemy.y], [ally.x, ally.y]) < (
                        self_radius + enemy_weapon_range + enemy_radius):
                    in_enemy_range = 1
                    break
                else:
                    in_enemy_range = 0
            if in_enemy_range:
                break

        north_bound, south_bound, west_bound, east_bound = self.calculate_distance_to_bounds(obs, for_subgroup=True)

        if self.previous_commands[self.current_group_id] == "FIGHT":
            prev_cmd = 1
        elif self.previous_commands[self.current_group_id] == "FLEE":
            prev_cmd = 0

        nw_enemy_presence, ne_enemy_presence, sw_enemy_presence, se_enemy_presence = self.detect_enemies_by_region(obs,
                                                                                                                   for_subgroup=True)

        distance_to_enemy = self.retrieve_distance_between_positions(self.retrieve_enemy_location(obs),
                                                                     self.get_avg_location_of_self_subgroup(obs))
        distance_to_enemy = distance_to_enemy / float((32 ** 2 + 20 ** 2) ** 0.5)

        return [current_hp, weapon_cooldown, enemy_in_range, in_enemy_range, prev_cmd, north_bound, south_bound,
                west_bound, east_bound,
                nw_enemy_presence, ne_enemy_presence, sw_enemy_presence, se_enemy_presence, self_unit_type,
                enemy_unit_type, self_weapon_range, enemy_weapon_range, self_speed, enemy_speed, distance_to_enemy]

    def detect_self_unit_types(self, obs):
        """
        detects the unit types of all currently available units as well as counts their initial numbers
        :param obs: pysc2 observation
        :return:list of unit type ids
        """
        feature_units = obs.observation.feature_units
        allies = [unit for unit in feature_units if unit.alliance == _PLAYER_SELF]

        unit_type_ids = []
        for ally in allies:
            if ally.unit_type not in unit_type_ids:
                unit_type_ids.append(ally.unit_type)

        self.unit_type_ids = unit_type_ids

        if self.first_move:
            for unit_type_id in unit_type_ids:
                self.init_unit_counts[unit_type_id] = 0
                for ally in allies:
                    if ally.unit_type == unit_type_id:
                        self.init_unit_counts[unit_type_id] += 1

        return unit_type_ids

    def select_current_unit_group(self, obs):
        """
        returns a pysc2 selection action for the unit group to be selected
        :param obs: pysc2 observation
        :return: pysc2 unitgroup selection action
        """
        self.is_select_step = False
        if len(self.unit_type_ids) == 0:
            return actions.FUNCTIONS.no_op()
        self.current_group_id = self.unit_type_ids[self.unit_group_select_counter % len(self.unit_type_ids)]
        units = self.get_units_by_type(obs, self.current_group_id)
        if len(units) > 0:
            unit = random.choice(units)
            self.unit_group_select_counter += 1
            return actions.FUNCTIONS.select_point("select_all_type", (unit.x, unit.y))

    def get_avg_location_of_self_subgroup(self, obs):
        """
        gets the average location of the currently selected unit subgroup
        :param obs: pysc2 obs
        :return: [position_x, position_y]
        """
        if self.current_group_id == 0:
            units = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]
        else:
            units = self.get_units_by_type(obs, self.current_group_id)

        if len(units) == 0:
            return self.latest_position
        player_x = []
        player_y = []
        for unit in units:
            player_x.append(unit.x)
            player_y.append(unit.y)
        pos = [int(sum(player_x) / len(player_x)), int(sum(player_y) / len(player_y))]
        self.latest_position = pos
        return pos

    def move_unit(self, obs, for_subgroup=False):
        """
        modified move_unit command from superclass. accounts for controlling subgroups instead of army
        :param obs: pysc2 observation
        :param for_subgroup:if being used for army or currently selected subgroup
        :return:as in super class
        """
        target = self.target
        if for_subgroup:
            current_location = self.get_avg_location_of_self_subgroup(obs)
        else:
            current_location = self.get_current_location(obs)
        dist_to_target = 2
        if ((abs(current_location[0] - target[0]) >= dist_to_target) or
                (abs(current_location[1] - target[1]) >= dist_to_target)):
            return {
                "function": actions.FUNCTIONS.Move_screen("now", (self.target[0], self.target[1])),
                "status": "MOVING_TO_TARGET"
            }
        else:
            return {
                "function": actions.FUNCTIONS.Move_screen("now", (self.target[0], self.target[1])),
                "status": "ARRIVED_AT_TARGET"
            }

    def reinitialize(self):
        self.fitness = 0
        self.target = [0] * 2
        self.obs = None
        self.latest_position = [0] * 2
        self.nn_output = []
        self.first_move = True
        self.step_counter = 1
        self.max_fitness = 0
        self.genome_threshold = 0
        self.current_group_id = 0
        self.unit_type_ids = []
        self.is_select_step = True
        self.unit_group_select_counter = 0
        self.enemy_id = 0
        self.previous_commands = {}
        self.init_unit_counts = {}

    def movement_step(self, distance, displacement, obs, for_subgroup=False):
        """Will have selected unit(s) move a specified distance in the
        directions cardinal directions, diagonals, and stay put
        :param distance: step size
        :param displacement: [x, y] in 0, 1 range to scale distance
        :param obs: pysc2 obs
        :param for_subgroup: if being used for army or currently selected subgroup
        """
        if for_subgroup:
            new_target = self.get_avg_location_of_self_subgroup(obs)
        else:
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

    def step(self, obs):
        # first step select army
        # second step send to engage
        # first move now finished
        # now alternate between selecting and controlling subgroups
        self.obs = obs

        self.detect_self_unit_types(obs)

        if self.train and np.random.rand() < 0.02:  # random exploration
            decision = np.random.randint(low=-1, high=1, size=1)
            displacement = [np.random.randint(low=-1, high=1, size=1), np.random.randint(low=-1, high=1, size=1)]
        else:
            decision = self.nn_output[0]
            # clamped or tanh (ALSO CHANGE BOOLEAN DECISION CHECK to 0)
            displacement = [self.nn_output[1], self.nn_output[2]]

        if self.first_move:
            if self.forced_engage:
                if self.can_do_action(obs, actions.FUNCTIONS.Move_screen.id):
                    move = self.move_unit(obs, for_subgroup=False)
                    if move["status"] is "ARRIVED_AT_TARGET":
                        self.first_move = False
                    else:
                        enemy_location = self.retrieve_enemy_location(obs)
                        current_location = self.get_current_location(obs)
                        if current_location[0] < enemy_location[0]:
                            target_location = [enemy_location[0] - 6, enemy_location[1]]
                        else:
                            target_location = [enemy_location[0] + 6, enemy_location[1]]
                        self.set_target_destination(target_location)
                        return self.move_unit(obs, for_subgroup=False)["function"]
                if self.can_do_action(obs, actions.FUNCTIONS.select_army.id):
                    return actions.FUNCTIONS.select_army("select")

        if self.is_select_step:
            if self.can_do_action(obs, actions.FUNCTIONS.select_point.id):
                self.is_select_step = False
                return self.select_current_unit_group(obs)
        else:
            if decision > 0:  # fight
                if self.can_do_action(obs, actions.FUNCTIONS.Attack_screen.id):
                    features = obs.observation.feature_units
                    enemies = [unit for unit in features if unit.alliance == _PLAYER_ENEMY]
                    if not enemies:
                        return actions.FUNCTIONS.no_op()
                    enemies = sorted(enemies, key=lambda x: x.health)
                    lowest_health_enemy_coords = [enemies[0].x, enemies[0].y]
                    self.is_select_step = True
                    self.previous_commands[self.current_group_id] = "FIGHT"
                    return actions.FUNCTIONS.Attack_screen("now", lowest_health_enemy_coords)
            else:  # flee
                if self.can_do_action(obs, actions.FUNCTIONS.Move_screen.id):
                    step_size = 10
                    self.movement_step(step_size, displacement, obs)
                    self.is_select_step = True
                    self.previous_commands[self.current_group_id] = "FLEE"
                    return actions.FUNCTIONS.Move_screen("now", (self.target[0], self.target[1]))
                    # move = self.move_unit(obs, for_subgroup=True)
                    # if move["status"] is "ARRIVED_AT_TARGET":
                    #     step_size = 10
                    #     self.movement_step(step_size, displacement, obs)
                    #     self.is_select_step = True
                    #     return actions.FUNCTIONS.Move_screen("now", (self.target[0], self.target[1]))
                    # else:
                    #     return move["function"]

        return actions.FUNCTIONS.no_op()

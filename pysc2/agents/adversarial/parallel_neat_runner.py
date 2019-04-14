# based on message based multiprocessing
# work in progress
# currently the environments spawn in subprocesses but the episode doesnt run properly. second step is last step

import os
import pickle

import time

import neat
from absl import app, flags
import multiprocessing
from multiprocessing import Process, Pipe
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import protocol
from pysc2.lib import actions

import visualize
from combat_agent import CombatAgent, HeteroCombatAgent, CombatAgent4
from movement_agent import MovementAgent
# from parallel_sc2_env import ParallelEnv
from units import units

TEST_ID = os.path.basename(__file__)

envs = []
agents = []
good_genomes = []

global_stats = None


class NeatNetworkType:
    FeedForward = 0
    Recurrent = 1


# command line arguments
params = flags.FLAGS

flags.DEFINE_string("testid", TEST_ID, "Test ID")
flags.DEFINE_integer("network_type", NeatNetworkType.FeedForward,
                     "Network types for NEAT evolution - 0 (Feedforward) or 1 (Recurrent)")
flags.DEFINE_integer("num_of_genome_eps", 1,
                     "Number of episodes to run per genome. Fitness is averaged across the episodes.")
flags.DEFINE_integer("num_of_agents", 1,
                     "Number of agents to parellelize training")
flags.DEFINE_string("checkpoint", None, "Checkpoint to resume training from")
flags.DEFINE_integer("generations", 100,
                     "Number of generations to run training for. Pass None to run until fitness threshold met or extinction occurs")
flags.DEFINE_string("config_filename", "config",
                    "Configuration filename for NEAT evolution")
flags.DEFINE_integer(
    "dimensions", 32, "Map Dimension size assuming square box map")
flags.DEFINE_string("map", "MoveToBeacon", "Name of SC2 Map to launch")
flags.DEFINE_integer("game_steps", 1000,
                     "Total number of game steps to run per episode")
flags.DEFINE_integer(
    "step_mul", 8, "Game steps made before agent step function called")
flags.DEFINE_boolean("visualize", False,
                     "To render training results in the PySC2 viewer")
flags.DEFINE_boolean("save_replay", False, "Save training replay")
flags.DEFINE_boolean(
    "parallel", False, "Parallelise training with multiple agents")
flags.DEFINE_float("genome_threshold", 1, "Score Threshold to save genome")
flags.DEFINE_boolean("simulate", False, "To simulate from pickle files")
flags.DEFINE_string("genome_file_path", "genomes.pkl",
                    "Path to pickle file for simulation")
flags.DEFINE_boolean("ensemble_control", False, "Use several networks to drive decisions")
flags.DEFINE_string("genome_file_type", "genome_list",
                    "Stats object or genome list in genome pickle - 'genome_list' or 'stats'")
flags.DEFINE_boolean("train", False, "To execute NEAT training")
flags.DEFINE_boolean("multinet", False, "Multiple nets for hetero control")

# important fix to allow spawned processes to create env and access absl flags
import sys

params(sys.argv)


def setup_neat_config(config_filename):
    # assumes config file is in same directory
    print("setting up config: ", config_filename)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_filename)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    return config


def plot_graphs(config, stats, display=False, winner=None):
    print("Plotting graphs")
    visualize.plot_stats(stats, ylog=False, view=display,
                         filename=params.testid + "-fitness.svg")
    visualize.plot_species(stats, view=display,
                           filename=params.testid + "-species.svg")
    if winner is not None:
        visualize.draw_net(config, winner, view=display,
                           node_names=None, filename=params.testid + "-net.svg")


def pickle_results(winner, genomes, stats):
    def pickle_dump(data, filename):
        with open(filename, 'wb') as output:
            pickle.dump(data, output, 1)

    print("Saving results to pickle files")
    pickle_dump(winner, params.testid + "-winner.pkl")
    pickle_dump(genomes, params.testid + "-genomes.pkl")
    pickle_dump(stats, params.testid + "-stats.pkl")


def eval_genomes(genomes, config):
    print("in eval genomes")
    global global_stats
    if params.parallel:
        print("Agents being used: {}".format(len(agents)))
        total_genomes = len(genomes)
        per_thread = int(total_genomes / params.num_of_agents)
        genome_splits = []
        for i in range(params.num_of_agents):
            genome_splits.append(genomes[i * per_thread:i * per_thread + per_thread])

        for i in range(params.num_of_agents):
            envs[i].parent_conn.send(("RUN", genome_splits[i]))

        results = []  # a list of dicts [genome.key, genome.fitness]
        for i in range(params.num_of_agents):
            msg, data = envs[i].parent_conn.recv()
            results.append(data)
            if msg == "DONE":
                continue

        print("Results received after evaluation from parallel environments")

        # collect and assign results
        # TODO find better way of combining results dictionaries
        result_1 = results[0]
        result_2 = {}
        # result_2 = results[1]
        combined_results = {**result_1, **result_2}
        for genome_id, genome in genomes:
            # assuming genome_id is the same as genome.key
            genome.fitness = combined_results[genome_id]

        print("Results assigned")


def close_envs(environments):
    print("Closing environments")
    for env in environments:
        env.parent_conn.send(("CLOSE", None))
        env.process.join()


def run_neat(config, agent_list, env_list, checkpoint):
    global global_stats

    if checkpoint is None:
        p = neat.Population(config)
        print("Starting a fresh NEAT Training using config file:{}".format(
            params.config_filename))
    else:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
        print("Resuming NEAT training using config file:{} from checkpoint:{}".format(
            params.config_filename, checkpoint))

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    global_stats = stats
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(time_interval_seconds=None,
                                     generation_interval=50, filename_prefix=params.testid + "-checkpoint"))

    winner = None
    try:
        winner = p.run(eval_genomes, n=params.generations)
    except KeyboardInterrupt or Exception:
        print("Training halted")
        plot_graphs(config, stats, display=False, winner=None)

    pickle_results(winner, good_genomes, stats)
    plot_graphs(config, stats, display=False, winner=winner)

    print("NEAT Training completed")


class ParallelEnv:
    def __init__(self, env_id, args):
        self.id = env_id
        self.args = args
        self.parent_conn = self.child_conn = None
        self.env = None
        self.agent = None
        self.config = None
        self.process = None
        self.network_type = NeatNetworkType.FeedForward
        self.num_of_genome_eps = args["num_of_genome_eps"]
        self.map = args["map"]
        self.dimensions = args["dimensions"]
        self.game_steps = args["game_steps"]
        self.step_mul = args["step_mul"]
        self.visualize = args["to_visualize"]

    def setup(self, agent, config):
        self.parent_conn, self.child_conn = Pipe()
        self.agent = agent
        self.config = config
        return self.parent_conn

    def start_env(self):
        args = (self.map, self.dimensions, self.game_steps
                , self.step_mul, self.visualize, self.config, self.agent)

        self.process = Process(target=self.run_process, args=args)
        self.process.start()

    def activate_network(self, obs, nn_input, net, agent):
        nn_output = net.activate(nn_input)
        agent.nn_output = nn_output
        print("nn_input", nn_input)
        print("nn_output", agent.nn_output)
        step_actions = [agent.step(obs)]
        print(step_actions)
        return step_actions

    def eval_single_genome(self, genome, config, agent, env):
        if self.network_type == NeatNetworkType.FeedForward:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
        elif self.network_type == NeatNetworkType.Recurrent:
            net = neat.nn.RecurrentNetwork.create(genome, config)

        agent.setup(env.observation_spec(), env.action_spec())

        fitness_total = 0

        print(env)

        for _ in range(self.num_of_genome_eps):
            timesteps = env.reset()

            print("genome eps ", self.num_of_genome_eps)

            agent.reset()

            obs = []
            nn_input = []
            fitness_episode = 0

            if agent.hetero:
                agent.sc2_env = env

            count = 0
            while True:
                count += 1
                print("here{} {}".format(count, multiprocessing.current_process().name))
                obs = timesteps[0]

                if obs.first():
                    agent.fitness_calculation_setup(obs)

                if agent.obs is None:
                    agent.obs = obs

                nn_input = agent.retrieve_inputs(
                    obs, CombatAgent.FeatureInputType.Handcrafted)

                step_actions = self.activate_network(obs, nn_input, net, agent)
                if obs.last():
                    # record_episode_ending(agent, obs)
                    print("last")
                    fitness_total += fitness_episode
                    break

                timesteps = env.step(step_actions)
                # print(timesteps[0])

                fitness_episode = agent.calculate_fitness(obs)

            agent.reinitialize()

        avg_fitness = float(fitness_total) / params.num_of_genome_eps

        return float(avg_fitness)

    def eval_genomes(self, env, genomes):
        results = {}
        for genome_id, genome in genomes:
            fitness_recorded = self.eval_single_genome(genome, self.config, self.agent, env)
            results[genome.key] = fitness_recorded
        return results

    def run_process(self, map_name, dimensions, step_mul, game_steps, to_visualize, config, agent):

        env = sc2_env.SC2Env(map_name=map_name,
                             players=[sc2_env.Agent(sc2_env.Race.terran)],
                             agent_interface_format=features.AgentInterfaceFormat(
                                 feature_dimensions=features.Dimensions(
                                     screen=dimensions, minimap=dimensions),
                                 use_feature_units=True),
                             step_mul=step_mul,
                             game_steps_per_episode=game_steps,
                             visualize=to_visualize,
                             random_seed=1)

        self.agent = CombatAgent4(
            id=0, map_width=dimensions, map_height=dimensions, train=False)

        self.agent.setup(env.observation_spec(), env.action_spec())

        timesteps = env.reset()
        self.agent.reset()
        count = 0
        while True:
            count += 1
            print("here{} {}".format(count, multiprocessing.current_process().name))
            obs = timesteps[0]

            if obs.first():
                self.agent.fitness_calculation_setup(obs)

            if self.agent.obs is None:
                self.agent.obs = obs

            # step_actions = self.activate_network(obs, nn_input, net, agent)
            step_actions = [actions.FUNCTIONS.no_op()]
            if obs.last():
                # record_episode_ending(agent, obs)
                print("last")
                break

            timesteps = env.step(step_actions)
            # print(timesteps[0])

        self.env = env
        # self.agent = agent
        # self.config = config

        self.child_conn.send("DONE")

        while True:
            msg, data = self.child_conn.recv()
            if msg == "RUN":
                # print(data)
                # import sys
                # print(sys.executable)
                results = self.eval_genomes(env, genomes=data)
                print(results)
                self.child_conn.send(("DONE", results))
            elif msg == "TEST":
                print("Here ", self.id)
                print(data)
            elif msg == "CLOSE":
                print("Closing environment")
                self.env.close()
                break


def setup_agents(num_of_agents, dimensions, is_training=False):
    print("Setting up agents")
    agent_list = [CombatAgent4(
        id=i, map_width=dimensions, map_height=dimensions, train=is_training) for i in range(0, num_of_agents)]
    print("Agents set up")
    return agent_list


def setup_envs(num_of_agents, map_name, dimensions, game_steps, step_mul, to_visualize, config, num_of_genome_eps):
    print("Setting up envs")
    global envs
    env_setup_params = {
        "map": map_name,
        "dimensions": dimensions,
        "game_steps": game_steps,
        "step_mul": step_mul,
        "to_visualize": to_visualize,
        "num_of_genome_eps": num_of_genome_eps
    }
    for i in range(num_of_agents):
        env = ParallelEnv(i, env_setup_params)
        # env.setup(agents[i], config)
        env.setup(None, config)
        envs.append(env)
        env.start_env()

    print(multiprocessing.current_process().name)

    for i in range(num_of_agents):
        msg = envs[i].parent_conn.recv()
        if msg == "DONE":
            continue

    print("Envs started successfully")

    time.sleep(1)
    #
    # test = ("TEST", [1, 2, 3])
    # for i in range(num_of_agents):
    #     envs[i].parent_conn.send(test)


def run():
    global agents, envs

    config = setup_neat_config(params.config_filename)

    # agents = setup_agents(params.num_of_agents, params.dimensions, params.train)
    setup_envs(params.num_of_agents, params.map, params.dimensions, params.game_steps, params.step_mul,
               params.visualize, config, params.num_of_genome_eps)

    run_neat(config, agents, envs, params.checkpoint)

    close_envs(envs)

    print("Clean termination")


def main(unused_args):
    run()


if __name__ == '__main__':
    app.run(main)

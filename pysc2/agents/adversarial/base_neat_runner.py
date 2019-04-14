import os
import pickle
import queue
import random
import threading
import time

import neat
from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import protocol

# functions to plot graphs and networks
import visualize

# import respective agent to use for training - must be plugged into setup_agents function
from combat_agent import CombatAgent, HeteroCombatAgent, CombatAgent4
from movement_agent import MovementAgent

# unit statistics taken from online wikis for information not available from pysc2
from units import units

# default test id
TEST_ID = os.path.basename(__file__)

# global variables for use in neat training
envs = []
agents = []

# used to store good results during training
good_genomes = []
global_stats = None


# used to initialise the type of networks to form from genomes
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
    "dimensions", 32, "Dimensions for feature layer observations from pysc2")
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
flags.DEFINE_float("genome_threshold", 1, "Manual Score Threshold to save genome. By default dynamically calculated")
flags.DEFINE_boolean("simulate", False, "To simulate from pickle files")
flags.DEFINE_string("genome_file_path", "genomes.pkl",
                    "Path to pickle file for simulation")
flags.DEFINE_boolean("ensemble_control", False, "Use several networks to drive decisions")
flags.DEFINE_string("genome_file_type", "genome_list",
                    "Stats object or genome list in genome pickle - 'genome_list' or 'stats'")
flags.DEFINE_boolean("train", False, "To execute NEAT training")
flags.DEFINE_boolean("multinet", False, "Multiple nets for hetero control")


def retrieve_agent_and_environment():
    """get an agent and environment from the global variables"""
    return agents.pop(), envs.pop()


def return_agent_and_environment(agent, env):
    """return an agent and environment to the global variables"""
    global agents, envs
    agents.append(agent)
    envs.append(env)
    return True


def activate_network(obs, nn_input, net, agent):
    """
    activate neural network using nn input, use the nn output to decide a pysc2 action in agent step function
    :param obs: pysc2 observation
    :param nn_input: list of inputs to be fed into the network
    :param net: net created from genome
    :param agent: agent being used for training/evaluation
    :return: a list of pysc2 actions to be sent into the environment
    """
    nn_output = net.activate(nn_input)
    agent.nn_output = nn_output
    step_actions = [agent.step(obs)]
    return step_actions


def ensemble_activation(nets, obs, nn_input, agent):
    """
    similar to the activate_network function but this time multiple networks are used to drive the networks
    this activation function is specific to the output definition [attack/flee, flee_x, flee_y] as is the case
    for the combat agents
    all the networks are activated and a fight or flee voting takes place
    a random network that voted for the winning decision is chosen to drive the next step
    :param nets: list of networks to be used for ensemble control
    :param obs: pysc2 observation
    :param nn_input: list of inputs to be fed into the network
    :param agent: agent being used for training/evaluation
    :return: a list of pysc2 actions to be sent into the environment
    """
    fight_voters = []
    flee_voters = []
    for net in nets:
        nn_output = net.activate(nn_input)
        # assuming output 1 is fight or flee
        if nn_output[0] > 0:
            fight_voters.append(net)
        else:
            flee_voters.append(net)
    if len(fight_voters) > len(flee_voters):
        choice_network = random.choice(fight_voters)
    elif len(fight_voters) < len(flee_voters):
        choice_network = random.choice(flee_voters)
    else:
        if random.randint(0, 1) == 0:
            choice_network = random.choice(fight_voters)
        else:
            choice_network = random.choice(flee_voters)
    nn_output = choice_network.activate(nn_input)
    agent.nn_output = nn_output
    actions = [agent.step(obs)]
    return actions


def save_good_genome(genome, fitness_threshold):
    """
    saves genome to global good genomes list if it passes a threshold value
    Note - can repeatedly save the same genome. does not check for uniqueness
    :param genome: genome to check fitness for
    :param fitness_threshold: fitness threshold to check against
    """
    global good_genomes
    if genome.fitness > fitness_threshold:
        good_genomes.append(genome)


def record_episode_ending(agent, obs):
    """
    function called at end of episode. Can be used to display or log useful information at users convenience
    :param agent: agent object at end of episode
    :param obs: last pysc2 observation for episode
    """
    print(agent.count_units(obs))


def eval_single_genome(genome, config, agent, env):
    """
    function to run an evaluation for a single genome
    :param genome: genome object to evaluate
    :param config: config object to be used to create net from genome
    :param agent: agent object to use for evaluation
    :param env: starcraft environment to be used for evaluation
    :return: float fitness value from evaluation
    """

    # create net from genome
    if params.network_type == NeatNetworkType.FeedForward:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    elif params.network_type == NeatNetworkType.Recurrent:
        net = neat.nn.RecurrentNetwork.create(genome, config)

    # setup agent
    agent.setup(env.observation_spec(), env.action_spec())

    # variable to accumulate fitnesses across number of episodes to run for evaluation
    fitness_total = 0

    # run evaluation for a given number of episodes for the same genome
    for _ in range(params.num_of_genome_eps):
        # refresh the sc2 environment for a new episode - timesteps[0] is the pysc2 observation
        timesteps = env.reset()

        # refresh the agent for a new episode
        agent.reset()

        obs = []
        nn_input = []
        fitness_episode = 0

        # loop for one episode
        while True:
            obs = timesteps[0]

            if obs.first():
                # perform any required setup on the first observation of the episode
                agent.fitness_calculation_setup(obs)

            if agent.obs is None:
                agent.obs = obs

            # obtain inputs from agent and observation
            nn_input = agent.retrieve_inputs(
                obs, CombatAgent.FeatureInputType.Handcrafted)

            # activate the network to receive a pysc2 action
            step_actions = activate_network(obs, nn_input, net, agent)
            if obs.last():
                # end of episode
                record_episode_ending(agent, obs)
                # add episode fitness to evaluation fitness
                fitness_total += fitness_episode
                break

            # step through the sc2 environment using the pysc2 action and obtain new observation
            timesteps = env.step(step_actions)

            # calculate the fitness
            fitness_episode = agent.calculate_fitness(obs)

        # cleanup and reinitialise the agent object for a new episode
        agent.reinitialize()

    # take an average of the fitnesses obtained across evaluation episodes
    avg_fitness = float(fitness_total) / params.num_of_genome_eps

    return float(avg_fitness)


class WorkerThread(threading.Thread):
    """worker thread for multithreaded genome evaluation"""

    def __init__(self, thread_id, agent_queue, env_queue, results, genomes, lock, config):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.agent_queue = agent_queue
        self.env_queue = env_queue
        self.results = results
        self.genomes = genomes  # list of genomes
        self.lock = lock
        self.config = config

    def run(self):
        agent = None
        env = None

        # each thread retrieves an agent and env, runs eval genome and returns back to queue
        self.lock.acquire()
        while True:
            if not self.agent_queue.empty() and not self.env_queue.empty():
                agent = self.agent_queue.get()
                env = self.env_queue.get()
                break
            else:
                # wait until an agent an env are available - timing may need tuning
                self.lock.release()
                time.sleep(0.1)
                self.lock.acquire()
        self.lock.release()

        # run a simulation with the acquired agent and env
        for genome_id, genome in self.genomes:
            if agent is not None and env is not None:
                result = eval_single_genome(genome, self.config, agent, env)
                genome.fitness = result
                self.results[genome.key] = result
                save_good_genome(genome, agent.genome_threshold)
            else:
                print("agent and env retrieved incorrectly")
                result = None

        self.lock.acquire()
        self.agent_queue.put(agent)
        self.env_queue.put(env)
        self.lock.release()


def eval_genomes(genomes, config):
    global global_stats
    if params.parallel:
        """
        MULTITHREADING - currently creating threads equal to number of agents and divides genomes between them
        speed increase caps out at 4 to 5 threads
        """
        # TODO environment timeout handling in rare case of environment failing
        thread_list = []
        queue_lock = threading.Lock()

        # create queues
        agent_queue = queue.Queue(maxsize=params.num_of_agents)
        env_queue = queue.Queue(maxsize=params.num_of_agents)
        results = {}
        queue_lock.acquire()
        for agent in agents:
            agent_queue.put(agent)
        for env in envs:
            env_queue.put(env)
        queue_lock.release()

        # create threads and divide genomes between them for evaluation
        thread_id = 1
        total_genomes = len(genomes)
        per_thread = int(total_genomes / params.num_of_agents)
        for i in range(params.num_of_agents):
            thread = WorkerThread(thread_id=thread_id, genomes=genomes[i * per_thread:i * per_thread + per_thread],
                                  agent_queue=agent_queue, env_queue=env_queue,
                                  lock=queue_lock, config=config, results=results)
            thread_list.append(thread)

        # if need to send genome to each thread
        # for genome_id, genome in genomes:
        #     thread = WorkerThread(thread_id=thread_id, genome=genome, agent_queue=agent_queue, env_queue=env_queue,
        #                           lock=queue_lock, config=config, results=results)
        #     thread_list.append(thread)
        #     thread_id += 1

        # start and join threads for one generation of genomes
        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()

        # collect and assign results
        for genome_id, genome in genomes:
            # assuming genome_id is the same as genome.key
            genome.fitness = results[genome_id]
    else:
        # serialised evaluation - env timeouts are handled
        agent, env = retrieve_agent_and_environment()
        for genome_id, genome in genomes:
            while True:
                try:
                    genome.fitness = eval_single_genome(genome, config, agent, env)
                    break
                except KeyboardInterrupt:
                    exit()
                except Exception as e:
                    print(e)
                    print("Exception during eval_single_genome")
                    if isinstance(e, protocol.ConnectionError):
                        print("timeout issues")
                        print("plotting graphs and pickling results")
                        print("attempting to restart env")

                    pickle_results([], good_genomes, [])
                    plot_graphs(config, global_stats, display=False, winner=None)

                    # close env
                    env.close()
                    # start new env
                    env = sc2_env.SC2Env(map_name=params.map,
                                         players=[sc2_env.Agent(sc2_env.Race.terran)],
                                         agent_interface_format=features.AgentInterfaceFormat(
                                             feature_dimensions=features.Dimensions(
                                                 screen=params.dimensions, minimap=params.dimensions),
                                             use_feature_units=True),
                                         step_mul=params.step_mul,
                                         game_steps_per_episode=params.game_steps,
                                         visualize=params.visualize,
                                         random_seed=1)
                    # reinitialise agent
                    agent.reinitialize()
                    # eval single genome should run again at top of while loop

            print("Genome ID {}, Genome fitness {}".format(
                genome_id, genome.fitness))

            save_good_genome(genome, agent.genome_threshold)
        return_agent_and_environment(agent, env)


def setup_neat_config(config_filename):
    """
    assumes config file is in same directory
    throws error if config file does not exist
    :param config_filename: name of filename
    :return: config object
    """
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_filename)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    return config


def print_run_info():
    print("Environment: {}".format("StarCraft2"))
    if params.train:
        print("Training with Parallelization: {}".format(params.parallel))
        print("Training using {} agents".format(params.num_of_agents))
        print("Training using {} episodes per genome".format(params.num_of_genome_eps))
        print("Training from checkpoint: {}".format(str(params.checkpoint)))
        print("Training to {} max generations".format(str(params.generations)))
    elif params.simulate:
        print("Running simulation using {} episodes per genome".format(params.num_of_genome_eps))
        print("Genome file being used: {}".format(params.genome_file_path))
        print("Using Ensemble control evaluation: {}".format(params.ensemble_control))
    print("Config file: {}".format(params.config_filename))
    print("Map being used: {}".format(params.map))


def plot_graphs(config, stats, display=False, winner=None):
    """
    plots graphs and networks for visualization
    -- graph of fitness against generations
    -- graph of species against generations
    -- network visualisations for winner
    :param config: config file to understand genomes
    :param stats: stats object tracks training statistics
    :param display: boolean to render display or just save to file
    :param winner:
    """
    print("Plotting graphs")
    visualize.plot_stats(stats, ylog=False, view=display,
                         filename=params.testid + "-fitness.svg")
    visualize.plot_species(stats, view=display,
                           filename=params.testid + "-species.svg")
    if winner is not None:
        visualize.draw_net(config, winner, view=display,
                           node_names=None, filename=params.testid + "-net.svg")


def pickle_results(winner, genomes, stats):
    """
    saves winner, genome and stats objects in pickle files
    :param winner: winner of neat evolution
    :param genomes: good genomes saved across training
    :param stats: stats object of neat evolution
    """

    def pickle_dump(data, filename):
        with open(filename, 'wb') as output:
            pickle.dump(data, output, 1)

    print("Saving results to pickle files")
    pickle_dump(winner, params.testid + "-winner.pkl")
    pickle_dump(genomes, params.testid + "-genomes.pkl")
    pickle_dump(stats, params.testid + "-stats.pkl")


def run_neat(config, agent_list, env_list, checkpoint=None):
    """
    main neat evolution runner
    creates genome population, launches genome evaluation, saves results and plots graphs
    :param config: config object to initialise and drive evolution
    :param agent_list: list of agent objects to be used in training
    :param env_list: list of sc2 env objects
    :param checkpoint: checkpoint to resume neat training from
    """
    global global_stats

    if checkpoint is None:
        p = neat.Population(config)
        print("Starting a fresh NEAT Training using config file:{}".format(
            params.config_filename))
    else:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
        print("Resuming NEAT training using config file:{} from checkpoint:{}".format(
            params.config_filename, checkpoint))

    # add statistics reporters to track neat evolution progress
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    global_stats = stats
    p.add_reporter(stats)

    # add checkpoint generator
    p.add_reporter(neat.Checkpointer(time_interval_seconds=None,
                                     generation_interval=50, filename_prefix=params.testid + "-checkpoint"))

    # run evaluation
    winner = None
    try:
        winner = p.run(eval_genomes, n=params.generations)
    except KeyboardInterrupt or Exception:
        print("Training halted")
        plot_graphs(config, stats, display=False, winner=None)

    # save results and plot graphs
    pickle_results(winner, good_genomes, stats)
    plot_graphs(config, stats, display=False, winner=winner)

    print("NEAT Training completed")


def setup_agents(num_of_agents, dimensions, is_training=False):
    """
    setup agents. respective agent must be imported from agent file
    :param num_of_agents: number of agents to setup
    :param dimensions: map dimensions
    :param is_training: if the agent is being used for training or evaluation
    :return: list of agents
    """
    print("Setting up agents")
    agent_list = [MovementAgent(
        id=i, map_width=32, map_height=20, train=is_training) for i in range(0, num_of_agents)]
    print("Agents set up")
    return agent_list


def setup_envs(num_of_envs, map_name, dimensions, game_steps, step_mul, to_visualize):
    """
    setup pysc2 envs
    :param num_of_envs: number of envs to setup
    :param map_name: map to launch
    :param dimensions: feature layer dimensions
    :param game_steps: number of game steps to run for each episode
    :param step_mul: number of game steps run for each observation
    :param to_visualize: boolean to render visualise
    :return: list of pysc2 envs
    """
    print("Setting up envs")
    environments = []
    for _ in range(0, num_of_envs):
        try:
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
            environments.append(env)
        except Exception as e:
            print(e)
    print("Envs set up")
    return environments


def save_replay(environments):
    """
    save replays sc2 replay directory which is printed
    filename of replay is the test id
    :param environments: sc2 envs
    """
    for env in environments:
        print(env.save_replay(params.testid))
        print("Replays saved to SC2 replay directory")


def close_envs(environments):
    print("Closing environments")
    for env in environments:
        env.close()


def ensemble_evaluation(genomes, config, agent, env):
    """
    used for ensemble evaluation of a list of genomes. uses the ensemble activation function
    :param genomes: list of genomes to use for ensemble control
    :param config: config object to drive evolution
    :param agent: agent object
    :param env: pysc2 env
    :return: average fitness obtained
    """
    nets = []
    for genome in genomes:
        if params.network_type == NeatNetworkType.FeedForward:
            nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        elif params.network_type == NeatNetworkType.Recurrent:
            nets.append(neat.nn.RecurrentNetwork.create(genome, config))

    agent.setup(env.observation_spec(), env.action_spec())

    fitness_total = 0

    for _ in range(params.num_of_genome_eps):
        timesteps = env.reset()

        agent.reset()

        obs = []
        nn_input = []
        fitness_episode = 0

        while True:
            obs = timesteps[0]

            if obs.first():
                agent.fitness_calculation_setup(obs)

            if agent.obs is None:
                agent.obs = obs

            nn_input = agent.retrieve_inputs(
                obs, CombatAgent.FeatureInputType.Handcrafted)

            step_actions = ensemble_activation(nets, obs, nn_input, agent)
            if obs.last():
                fitness_total += fitness_episode
                break
            timesteps = env.step(step_actions)

            fitness_episode = agent.calculate_fitness(obs)

        agent.reinitialize()

    avg_fitness = float(fitness_total) / params.num_of_genome_eps

    return float(avg_fitness)


def multinet_eval_single_genome(genome_1, genome_2, config, agent, env):
    """
    sample evaluation method for multinet control using already trained networks with the hetero agent. need to modify
    inputs according to genome training and config
    hardcoded genome to net conversion TODO need to update to dynamic through list of genomes and unit type
    :param genome_1: genome 1 corresponding to stalkers
    :param genome_2: genome 2 corresponding to hellions
    :param config: config object
    :param agent: agent object
    :param env: pysc2 env
    :return: average evaluated fitness
    """
    nets = {}
    nets[units["STALKER"]] = neat.nn.FeedForwardNetwork.create(genome_1, config)
    nets[units["HELLION"]] = neat.nn.FeedForwardNetwork.create(genome_2, config)

    agent.setup(env.observation_spec(), env.action_spec())

    fitness_total = 0

    for _ in range(params.num_of_genome_eps):
        timesteps = env.reset()

        agent.reset()

        obs = []
        nn_input = []
        fitness_episode = 0

        while True:
            obs = timesteps[0]

            if obs.first():
                agent.fitness_calculation_setup(obs)

            if agent.obs is None:
                agent.obs = obs

            nn_input = agent.retrieve_inputs(
                obs, CombatAgent.FeatureInputType.Handcrafted)

            if agent.current_group_id is not 0:
                print(agent.current_group_id)
                net = nets[agent.current_group_id]
            else:
                net = nets[units["HELLION"]]
            step_actions = activate_network(obs, nn_input, net, agent)
            if obs.last():
                fitness_total += fitness_episode
                break
            timesteps = env.step(step_actions)

            fitness_episode = agent.calculate_fitness(obs)

        agent.reinitialize()

    avg_fitness = float(fitness_total) / params.num_of_genome_eps

    return float(avg_fitness)


def multinet_simulation(config, agents, envs):
    """load and run genomes for multinet evaluation"""
    genome_1 = None
    genome_2 = None

    agent, env = retrieve_agent_and_environment()

    # stalker
    with open('final-winner_sz.pkl', 'rb') as genomes_file:
        genome_1 = pickle.load(genomes_file)
        print(genome_1.fitness)

    # hellion
    with open('final-winner_hz.pkl', 'rb') as genomes_file:
        genome_2 = pickle.load(genomes_file)
        print(genome_2.fitness)

    multinet_eval_single_genome(genome_1, genome_2, config, agent, env)
    return_agent_and_environment(agent, env)


def simulate_genomes(config, agents, envs, genome_file_path, genome_file_type="genome_list", ensemble_control=False):
    """
    to simulate the performance of a trained genome model
    :param config: config object to create nets
    :param agents: agents to use for evaluation - must correspond to genome training agents
    :param envs: pysc2 envs
    :param genome_file_path: relative path of genome pickle file
    :param genome_file_type: 'genome_list' or 'stats' object
    :param ensemble_control: boolean for ensemble evaluation
    """
    genomes = []

    with open(genome_file_path, 'rb') as genomes_file:
        pickle_load = pickle.load(genomes_file)

    if genome_file_type == "stats":
        genomes = pickle_load.best_unique_genomes(3)
    else:
        genomes = pickle_load
        if type(genomes) is not list:
            genomes = [genomes]

    print("{} Genomes loaded for simulation".format(str(len(genomes))))

    agent, env = retrieve_agent_and_environment()

    if ensemble_control:
        # assume stats object being used
        print("Simulating with ensemble control")
        if len(genomes) > 3:
            # pickup first three if number of genomes too big 
            genomes = genomes[:2]
        fitness_recorded = ensemble_evaluation(genomes, config, agent, env)
        print("Fitness achieved with ensemble control: {}".format(fitness_recorded))
        print("Genomes used:")
        for genome in genomes:
            print("Genome id {}".format(genome.key))
    else:
        for genome in genomes:
            genomes = sorted(genomes, key=lambda x: x.fitness, reverse=True)
            fitness_recorded = eval_single_genome(genome, config, agent, env)
            print("Genome id {}, Genome fitness {}".format(genome.key, fitness_recorded))
            visualize.draw_net(config, genome, view=True,
                               node_names=None, filename=params.testid + "-net.svg")

    print("Pickle Simulation completed")
    return_agent_and_environment(agent, env)


def run():
    global agents, envs
    print_run_info()

    config = setup_neat_config(params.config_filename)

    agents = setup_agents(params.num_of_agents, params.dimensions, params.train)
    envs = setup_envs(params.num_of_agents, params.map, params.dimensions,
                      params.game_steps, params.step_mul, params.visualize)

    if not params.simulate:
        run_neat(config, agents, envs, params.checkpoint)
    else:
        print("Running genome simulation")
        if params.multinet:
            multinet_simulation(config, agents, envs)
        else:
            simulate_genomes(config, agents, envs, params.genome_file_path, params.genome_file_type)

    if params.save_replay:
        save_replay(envs)

    close_envs(envs)

    print("Clean termination")


def main(unused_args):
    run()


if __name__ == "__main__":
    app.run(main)

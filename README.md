# Neuroevolution for Starcraft 2 Micromanagement through NEAT

NEAT applied to StarCraft 2 micromanagement scenarios. Evolved agents learn to perform hit and run/ kiting strategies
in ranged vs melee matchups. An extensive list of maps is given in pysc2/agents directory.

## Install
Install dependencies while in the root directory:
`pipenv install`

Make sure you have pipenv installed. Else run `pip install pipenv`

You need to have python3 installed.

## Framework

The `base_neat_runner.py` is a generic framework to allow the quick training and simulation of neuroevolutionary agents.

To run: `python base_neat_runner.py`

The above command accepts command line arguments:

| Argument flag | Description |
| --- | --- |
| `--map` | Name of SC2 Map to launch |
| `--num_of_genome_eps` | Number of episodes to run per genome. Fitness is averaged across the episodes. |
| `--testid` | name for test run and result file prefix |
| `--train` | To execute NEAT training |
| `--network_type` | Network types for NEAT evolution - 0 (Feedforward) or 1 (Recurrent) |
| `--num_of_agents` | Number of agents to parellelize training |
| `--checkpoint` | Checkpoint to resume training from |
| `--generations` | Number of generations to run training for. Pass None to run until fitness threshold met or extinction occurs |
| `--config_filename` | Configuration filename for NEAT evolution. Relative path to neat runner file |
| `--dimensions` | Dimensions for feature layer observations from pysc2 |
| `--game_steps` | Total number of game steps to run per episode |
| `--step_mul` | Game steps made before agent step function called |
| `--visualize` | To render training results in the PySC2 viewer |
| `--save_replay` | Save training replay |
| `--parallel` | Parallelise training with multiple agents |
| `--genome_threshold` | Manual Score Threshold to save genome. By default dynamically calculated |
| `--simulate` | To simulate from pickle files |
| `--genome_file_path` | Path to pickle file for simulation |
| `--ensemble_control` | Use several networks to drive decisions |
| `--genome_file_type` | Stats object or genome list in genome pickle - 'genome_list' or 'stats' |
| `--multinet` | Multiple nets for hetero control |


Parellisation is allowed with multithreading.

Parellelisation with multiprocessing is a work in progress with the `parallel_neat_runner.py`

## Generic Agent

The `base_agent.py` is a generic neat agent that is built to work with the neat runner.
Any custom agents can extend the `BaseNeatAgent` class to be able to run with the framework. Then for customised
behaviour the new agent class only needs to override the following methods: `step`, `calculate_fitness` and the 
`retrieve_handcrafted_inputs` or `retrieve_pixel_inputs`. An example of this functionality is the `movement_agent.py`
or the `combat_agent.py`

## Combat Agent

`combat_agent.py` has a number of agents implemented built for the maps in our map directory. They use the NEAT paradigm
and differ mainly in their input selection, fitness calculation and step control logic

`CombatAgent4` successfully learnt kiting strategies on the 5v5 Hellions vs Zealots matchup using the
 `combat_test_4-config3` configuration file

## Results

https://i.cs.hku.hk/fyp/2018/fyp18045/

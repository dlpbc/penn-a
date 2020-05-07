## PENN-A

Plastic Evolved Neuromodulated Network with Autoencoder (PENN-A).
Code for the paper [Evolving Inborn Knowledge For Fast Adaptation in Dynamic POMDP Problems](#)

## Dependencies
#### conda (from exported environment)
```
conda env create -f environment.yml
conda activate py36
```

#### pip
1. install [graphviz](https://www.graphviz.org/) and add the binaries to system path.
2. install the dependencies via pip
```
pip installl -r requirements.txt
```

### CT-graph environment
~~For now, the CT-graph environment is in use internally (and it may still undergo further development). It should be public available in the future. If you wish to get access to it, please feel free to contact me.~~

The CT-graph environment has been added to this repository. Installation instructions below.
```
cd ctgraphenv
pip install -r requirements.txt
pip install -e .
```

### Malmo Minecraft environment
Follow the installation procedure from the official repo [link](https://github.com/Microsoft/malmo/tree/master/MalmoEnv)
Note: follow the installation procedure for **MalmoEnv** which is the preferred way going forward. Do not follow the installation procedure **Malmo as a native wheel**.

## Usage
### CT-graph Environment
1. To launch a sample experiment with this environment, run the command below
```
python train_ctgraph.py -g 10 -p 20 -e 10
```

2. For help, run the command below
```
python train_ctgraph.py -h
```

3. To launch an experiment based on configurations from the paper, run the command below
```
python train_ctgraph.py -g 200 -p 600 -e 100 --num-workers 8 --env-config-path ./envs_config/ctgraph/graph_depth2.json
```

4. To launch an experiment based on configurations from the paper, run the command below
```
python train_ctgraph.py -g 200 -p 800 -e 100 --num-workers 8 --env-config-path ./envs_config/ctgraph/graph_depth3.json
```

5. To test a trained agent, use the command below as a template, providing the required paths
```
python test_ctgraph.py log/path/to/exp/config log/path/to/saved/agent log/path/to/feat/extractor -e 100
```

### Minecraft Environment
Pre-usage (optional):
If headless run is preferred, first run the command below before starting minecraft
 `xvfb-run -s "-screen 0 1400x900x24" /bin/bash`

1. start minecraft (navigate to installation folder /path/to/malmo/Minecraft, and then run the command
 `./launchCient.sh -port 9000 -env`
Note:  if --mc-resync will be set to a value > 0 in the python train script command below, then make sure that minecraft malmo would have been started with the `-replaceable` flag included. For example
 `./launchClient.sh -port 9000 -env -replaceable`

2. now launch experiment (for example)
```
python train_minecraft.py -g 10 -p 20 -e 20 --mc-mission envs_config/minecraft/env_maze_double_tmaze.xml --mc-goals envs_config/minecraft/env_maze_double_tmaze_goals.json --mc-resync 0 

or

python train_minecraft.py -g 10 -p 20 -e 20 --mc-mission envs_config/minecraft/env_maze_double_tmaze.xml --mc-goals envs_config/minecraft/env_maze_double_tmaze_goals.json --mc-resync 500 
```

#### Minecraft Environment Instance Replica
The specific minecraft environment instance (double t-maze) used for this work was replicated independent of the minecraft engine. The major goal for this was to speed up the experiments, as the client/server setup of the default minecraft environment lead to slow experiments. Furthermore, with the client/server setup made it less trivial to run multiple environments in parallel for agent evaluation (useful for further speed up). The replica environment instance is lightweight and can be easily used for parallel evaluation of agents. It is important to note that the the replica environment is only for the specific minecraft environment instance, and not configurable to build different environments like the default minecraft.

1. To launch a sample experiment with this environment, run the command below
```
python train_mcaltenv.py -g 10 -p 20 -e 20 --num-workers 4
```
2. To launch an experiment based on configurations from the paper, run the command below
```
python train_mcaltenv.py -g 400 -p 800 -e 50 -t 8 --num-workers 4
```
3. For help, run the command below
```
python train_mcaltenv.py -h
```
4. To test a trained agent, use the command below as a template, providing the required paths
```
python test_mcaltenv.py log/path/to/exp/config log/path/to/saved/agent log/path/to/feat/extractor -e 100
```

### Disclaimer
The codebase was refactored and cleaned up (some additions and deletions) before public release. If you encounter any issues with the code or spot a bug, please feel free to contact me (for example, open a github issue in this repository). Feedback is welcomed.

### Credits for baselines
Publicly available codebase (with some modification) was employed to conduct experiments for baseline methods.
Thank you to the author(s) of the repositories.
- CAVIA/MAML: <https://github.com/lmzintgraf/cavia>
- RL^2: <https://github.com/jonasrothfuss/ProMP> (full code branch)

### Citation (BibTex)
```

```

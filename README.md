# Isaaclab Experiments 

This repository has many research projects with RL on Isaaclab. It is designed to be modular, so each folder inside 
isaaclab_experiments has its own files for running the project.


# Getting Started

After cloning this repository, run this command: 
```bash 
git submodule init
```

To install Isaac Lab, follow the pip installation from the original repository: [https://isaac-sim.github.io](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).

During the official instalation, you are prompted for cloning Isaac Sim, so just use the folder `IsaacLab` in this repository instead.


After you are done with this setup, install this repository with the following command:
```bash
pip install -e .
```

To run an experiment for training:

```bash
python scripts/train.py --headless --task Go1-Navigation-v0
```

To view training logs

```bash 
tensorboard --logdir logs
```

# How the code is organized

All experiments are inside the folder `isaaclab_experiments`, take a look at the folder `go1_low_level` with an example of the possible parts.

# Justfile

We included a Justfile for running the train and play commands for the available environments.

**Install justfile utility**: https://github.com/casey/just

**Setup autocomplete**: 

```
echo 'source <(just --completions bash)' >> ~/.bashrc
source ~/.bashrc
```


Now you can enjoy autocompletions: 
```
just train<tab> 
 ```
```
```

If you add a new environment, regenerate the justfile:
```
just generate_envs
```

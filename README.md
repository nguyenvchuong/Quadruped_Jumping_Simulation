# Learning
This is the master branch for quadruped simulation.

## Installation

Recommend using a virtualenv with python3.8 (tested). Recent pybullet (3.2.2), gym(0.17.0), pip (21.3.1)  numpy (1.24.2), etc.

## Code structure

- [imitation_tasks](./usc_learning/imitation_tasks) for any trajectories from the optimization, and classes to compare current robot state with desired ones, etc.
- [envs](./usc_learning/envs) any robot classes and basic gym environments. See in particular [quadruped_master](./usc_learning/envs/quadruped_master) for a general interface for aliengo and laikago.

## How to run the code

- For jumping, we use usc_learning/imitation_tasks/main.py

## Useful command line
- to delete .git file via terminal, we can use rm -fr .git
- to delete file, rm -f filename
- to delete folder, rm -r foldername

## TODO


# Magnetization Simulation using DQN

This project simulates magnetization dynamics using a Reinforcement Learning approach (Deep Q-Networks, DQN). The simulation models the time evolution of magnetization under the influence of external magnetic fields and anisotropy, and the agent learns to control the magnetic field to optimize the magnetization reversal process.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
- [Output](#output)
- [Dependencies](#dependencies)
- [Results](#results)

## Overview

This project simulates magnetization dynamics with the goal of optimizing the reversal of magnetization using reinforcement learning. The magnetization is influenced by external fields, shape anisotropy, and damping, and a DQN agent learns to control the external magnetic field to achieve optimal magnetization reversal.

## Features

- Simulates magnetization dynamics using Runge-Kutta methods.
- Deep Q-Learning agent to control magnetic field variations.
- Visualization of magnetization and magnetic field evolution over time.
- Saves simulation data and plots for detailed analysis.

## Usage

To run the magnetization simulation with the DQN agent, execute hte `main.py` file:

```bash
python main.py
```

### Parameters

You can modify the following key parameters in the `main()` function:
- **episodes**: Number of episodes for tarining the DQN agent.
- **t_limit**: Total simulation time.
- **dt**: Time step for the simulation.
- **alphaG**: Gilbert damping constant.
- **anisotropy**: Anisotropy field vector.
- **H_shape**: Shape magnetic field.
- **dh**: Magnitude of magnetic field variation per action.
- **da**: Time between actions.
- **m0**: Initial magnetization direction.

## Output

During the simulation, the program will generate the following outputs:
 1. **Reward History**: A file `reward_history.txt` containging the rewards achieved by the DQN agent over the episodes.

 2. **Magnetization and Field Data**: Files `m.txt`, `h.txt`, and `t.txt` storing the magnetization, magnetic field, and time evolution data for the best episode.

 3. **Plots**:
    - `reversal_time.png`: A plot of magnetization reversal time.
    - `field.png`: A plot of external, anisotropy, and shape magnetic fields over time.
   
The best episode's data and other results are saved in a directory named based on the simulation parameters (e.g., `H=x_dh=100_da=0.01_ani=(0,0,100)`).

## Dependencies

The following Python libraries are required:
- `numpy`
- `matplotlib`
- `toch`
- `deque`
- `random`
- `os`
- `copy`
- `datetime`

## Results





# Reinforcement Learning Program for Estimating External Magnetic Field Application Methods for High-Speed Magnetization Reversal

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
- [Output](#output)
- [Dependencies](#dependencies)
- [Results](#results)

## Overview

This project simulates magnetization dynamics using a Reinforcement Learning approach (Deep Q-Networks, DQN). The simulation models the time evolution of magnetization under the influence of external magnetic fields and anisotropy, and the agent learns to control the magnetic field to optimize the magnetization reversal process. In this context, optimization refers to reversing the magnetization as quickly as possible while ensuring it stops after the reversal is completed. 

## Features

- Simulates magnetization dynamics using Runge-Kutta methods.
- Deep Q-Learning agent to control magnetic field variations.
- Visualization of magnetization and magnetic field evolution over time.
- Saves simulation data and plots for detailed analysis.

## Usage

To run the magnetization reversal simulation with the DQN agent, execute one of the following Python files depending on the magnetic field configuration:
 - For simulations with a one-directional magnetic field (along the x-axis), run `x.py`:

   ```bash
   python x.py
   
 - For simulations with a two directional magnetic field (along the x and y axes), run `xy.py`:
   ```bash
   python xy.py

### Directory structure

To run the program, ensure that the `modules` directory is present in the project root. This contains essential modules required for numerical calculations of magnetization dynamics and graphical plotting of results.

```bash
/project_root
│
├── x.py
├── xy.py
└── modules/
    ├── plot.py
    └── system.py
```

### Parameters

You can modify the following key parameters in the `main()` function:
- **episodes**: The number of episodes to run the simulation.
- **t_limit**: The time limit for each episode (in seconds).
- **alphaG**: Gilbert damping constant.
- **anisotropy**: The magnetic anisotropy constant (in Oersted). A positive value indicates an easy axis of magnetization, while a negative value indicates a hard axis.
- **H_shape**: The influence of the demagnetizing field (in Oersted).
- **dh**: Magnitude of magnetic field variation per action (in Oersted).
- **da**: The action interval (in seconds).
- **m0**: The initial magnetization vector.
- **directory**: The name of the directory where the results are saved.

## Output

During the simulation, the program will generate the following outputs:
 1. **Reward History**: A file `reward_history.txt` containging the rewards achieved by the DQN agent over the episodes.

 2. **Magnetization and Field Data**: Files `m.txt`, `h.txt`, and `t.txt` storing the magnetization, magnetic field, and time evolution data for the best episode.

 3. **Plots**:
    - `reversal_time.png`: A plot of magnetization reversal time.
    - `field.png`: A plot of external, anisotropy, and shape magnetic fields over time.

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





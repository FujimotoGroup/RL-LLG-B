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

The program generates several output files in the specified directory:

- `options.txt`: Contains the parameters used in the simulation, as well as the magnetization reversal time.
- `episodeXXXXX.png`: Graphs of the magnetic field and magnetization for the corresponding episode.
- `episode00000.png`: Graphs of the magnetic field and magnetization for the optimal policy episode. :warning:
- `reversal_time.png`: Shows the graph of magnetization under the optimal policy with a tangent line that defines the magnetization reversal time.
- `field.png`: Plots the external field, anisotropy field, and demagnetizing field over time.
- `m.txt`: Contains the magnetization values over time during the optimal policy episode.
- `h.txt`: Contains the external magnetic field values over time during the optimal policy episode.
- `t.txt`: Contains the time values used in the simulation during the optimal policy episode.
- `reward history.txt`: Tracks the reward history of the DQN agent across episodes.

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

The results of the simulations are stored in two directories: `./Results` and `./Prior_Research`.

- **`./Results`**: This directory contains the simulation results based on the parameters we set independently. The directory names reflect the parameters used for the simulations, such as `./Results/H=x_dH=100_da=0.1_ani=(0,0,100)`, where `H=x` indicates the direction of the external magnetic field, `dH=100` refers to the magnetic field change per action (in Oersted), `da=0.1` represents the interval between actions (in nanoseconds), and `ani=(0,0,100)` denotes the magnetic anisotropy constants (in Oersted). The contents of each directory match the outputs described in the [Output](#Output) section.

- **`./Prior_Research`**: This directory contains the results of simulations using the same parameters as those found in prior research. Inside, there are two subdirectories: `Bauer` and `Schumacher`, each referring to specific studies:
  - `./Prior_Reseach/Bauer`: M. Bauer, J. Fassbender, B. Hillebrands, and R. L. Stamps, "Switching behavior of a Stoner particle beyond the relaxation time limit," Phys. Rev. B **61**, 3410 (2000).
  - `./Prior_Reseach/Schumacher`: H. W. Schumacher, C. Chappert, R. C. Sousa, P. P. Freitas, and J. Miltat, "Quasiballistic Magnetization Reversal," Phys. Rev. Lett. **90**, 017204 (2003).



# Cyclesgym

This repository contains an [OpenAI gym](https://gym.openai.com/) interface to the [Cycles 
crop growth simulator](https://plantscience.psu.edu/research/labs/kemanian/models-and-tools/cycles).

For more information about cyclesgym, see our [user manual](documents/manual.md).

## Installation

We recommend Python 3.8+ installation using [Anaconda](https://www.anaconda.com/products/individual#downloads).

First, create and activate a virtual environment using Anaconda:

```bash
conda create -yn cyclesgym python=3.8
conda activate cyclesgym
```

Then, clone the repo and change working directory

```bash
git clone https://github.com/kora-labs/cyclesgym.git
cd cyclesgym
```

Subsequently, install the library according to your needs.
To install, run:

```bash
pip install -e .
```

If you further want to use some basic libraries to train reinforcement learning agents of the cyclesgym environments use:
```bash
pip install -e .SOLVERS
```

Or, if you are using zsh:
```bash
pip install -e .\[SOLVERS\]
```

## Contextual MDP experiments

We provide a set of scripts to run contextual MDP experiments on the cyclesgym environments with crop planning tasks.

To train the policy and evaluate on CMDP, run the following command:
```bash
cd experiments/crop_planning
sbatch s-crop-train-transfer.sh
```
For multi-task training, run:
```bash
sbatch s-crop-train-transfer-multi.sh
```
The results will be saved in the `results` directory.


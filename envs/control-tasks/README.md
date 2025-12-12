# Control Tasks (CartPole and BipedalWalker)

Contextual MDP experiments on CARL control tasks to check the training instability and brittleness to the task variation.

## Installation

### Environment

Before running the experiments, you need to install the required dependencies. You can do this by creating a new conda environment with the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate carl
```

This experiments lay on top of the [CARL](https://github.com/automl/CARL) library. You can install it in the parent directory by running:

```bash
cd ..
git clone https://github.com/automl/CARL.git --recursive
cd CARL
git checkout b5e12d20cf3f9fe3450a27875867754a78e404d7
git submodule update --init --recursive
pip install .
```

This will only install the basic classic control environments, which should run on most operating systems. For the full set of environments, use the install options:

```bash
pip install -e .[box2d,brax,dm_control]
```

Return to the `control-tasks` directory:

```bash
cd control-tasks
```

## Running the experiments

For CartPoleCMDP with three variables, you can run the following command if you are using a cluster with SLURM:

```bash
sbatch cartpole_train_3d.sh
```
and transfer the learned policy to the whole CartPoleCMDP task:

```bash
sbatch cartpole_transfer_3d.sh
```

For BipedalWalkerCMDP, you can run the following command (train and then transfer):

```bash
sbatch walker_train_transfer_3d.sh
```

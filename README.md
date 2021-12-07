# gym-STED-pfrl

This repository contains the experiments used to train a reinforcement learning (RL) agent in the `gym-sted` environment. Please refer to the `gym-sted` repository to obtain specific information.

## Description

This repository relies heavily on the [PFRL](https://github.com/pfnet/pfrl) deep learning library. You can also access the original [README.md](https://github.com/FLClab/gym-sted-pfrl/blob/main/PFRL-README.md) from PFRL to get more information about the library.

## Installation

We recommend that the user installs this version of the library. To perform the installation follow the next steps. The installation was tested with Python 3.7.

```bash
git clone https://github.com/FLClab/gym-sted-pfrl
pip install -e gym-sted-pfrl
```

The current library also relies on the installation of `gym-sted` and `pysted`. Please refer, to their specific repository for more information.

## Usage

In the following section, we provide specific details on how a user may reproduce the experiments that were done in the publication.

### Train

To train an agent, the user may use the `main.py` file which is provided in this repository with specific options. Please use the `--help` option for details.

For example, the user may launch a short training using

```bash
python main.py --env gym_sted:ContextualMOSTED-easy-v0 --eval-n-runs 2 --steps 10 --eval-interval 5 --gamma 0. --exp-id debug
```

We also provide the `bash` files that were used to train our models in [bash-files](https://github.com/FLClab/gym-sted-pfrl/blob/main/bash-files).

### Evaluation

To evaluate an agent, the user may use the `run_gym1.py` file which is provided in this repository in the `src/analysis` folder.

For example, the user may evaluate the model that was trained above by using

```bash
cd ./src/analysis
python run_gym1.py --model-name debug --num-envs 1 --eval-n-runs 2
```

We also provide the `bash` files that were used to evaluate our models in [bash-files](https://github.com/FLClab/gym-sted-pfrl/blob/main/bash-files/eval).

### Manual interaction

We provide an example file in which a user may interact with the environment by sequentially choosing the actions. The acquired images and history of parameters/objectives are presented to the user at each time step.

To test the manual interaction, the user may launch the following

```bash
python human.py
```

## Citation

If you use the experiment files provided please cite us
```bibtex
@misc{turcotte2021pysted,
  title = {pySTED : A STED Microscopy Simulation Tool for Machine Learning Training},
  author = {Turcotte, Benoit and Bilodeau, Anthony and Lavoie-Cardinal, Flavie and Durand, Audrey},
  year = {2021},
  note = {Accepted to AAAI 2021}
}
```

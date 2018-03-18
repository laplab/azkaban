# Azkaban 🏯
Multiagent environments for Reinforcement Learning with emphasis on communication

## Installation
Azkaban relies on Python 3 and PyTorch. Please follow the [instruction](http://pytorch.org/) to install the second one manually.  
Other dependencies can be installed as usual:

```bash
pip install -r requirements.txt
python setup.py install
```

## Examples
Examples of making experiments with Azkaban are located in [notebooks](https://github.com/laplab/azkaban/tree/master/notebooks) folder.

## Structure
```
├── azkaban
│   ├── agent
│   │   ├── a3c.py          # A3C agent
│   │   ├── qlearning.py    # Tabular Qlearning agent
│   │   └── stochastic.py   # Random agent
│   ├── display             # utilities for visualizing env state
│   ├── env
│   │   └── team.py         # Teams environment
│   ├── monitor             # utilities for recording sessions
│   ├── optim
│   │   └── shared_adam.py  # Adam for async use by multiple agents
│   ├── space
│   │   └── discrete.py     # Discrete action space
│   └── utils               # code utilities
├── docs                    # documentation and experiments logs
└── notebooks               # notebooks with experiments
```

## Goals
Experiments results can be seen [here](https://laplab.github.io/azkaban/). Official university project description can 
be found [here](http://wiki.cs.hse.ru/Learning_communicative_and_cooperative_strategies_in_multi-agent_decision_processes_(проект)) (russian).

- [x] Environment
- [x] Random agent
- [x] Tabular Qlearning agent
- [x] A3C agent
- [ ] Basic communication
- [ ] [maddpg](https://blog.openai.com/learning-to-cooperate-compete-and-communicate/)

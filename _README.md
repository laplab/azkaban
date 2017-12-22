# Azkaban

```python
from azkaban.agent import RandomAgent, GreedyAgent
from azkaban.env import TeamsEnv, TeamsEnvConf
from azkaban.monitor import VideoMonitor

# 5x5 field
conf = TeamsEnvConf(
    world_shape=(5, 5),
    comm_shape=(5,)
)

# 1 team - agents acting randomly
# 2 team - agents greedily looking for someone to kill
env = TeamsEnv(
    teams=[
        (RandomAgent(conf=conf), RandomAgent(conf=conf)),
        (GreedyAgent(conf=conf), GreedyAgent(conf=conf))
    ],
    conf=conf
)

# wrap env with monitor which will record play session as a video
env = VideoMonitor(env)
env.reset(outfile='result.mp4')

done = False
for i in range(1000):
    done = env.step()

    if done:
        break

# finish recording if there is no winner
# otherwise it will be finished automatically
if not done:
    env.finish()
```

## Notes
To prevent `matplotlib` from creating windows while recording, use headless backend:

```python
import matplotlib

matplotlib.use('Agg')
```
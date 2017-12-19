# Azkaban

```python
from azkaban.agent import RandomAgent
from azkaban.env import TeamsEnv, TeamsEnvConf
from azkaban.monitor import VideoMonitor

# create 2-teams env with agents acting randomly
env = TeamsEnv(
    teams=[(RandomAgent(), RandomAgent()), (RandomAgent(), RandomAgent())],
    conf=TeamsEnvConf(
        world_shape=(5, 5),
        comm_shape=(5,)
    )
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
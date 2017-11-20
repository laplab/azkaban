import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from azkaban.monitor.core import Monitor


class VideoMonitor(Monitor):
    def __init__(self, env, fps=15, dpi=300, writer_type='ffmpeg'):
        self.env = env

        self.writer = None
        self.outfile = None
        self.fps = fps
        self.dpi = dpi
        self.writer_type = writer_type

    def step(self):
        done = self.env.step()
        self.env.render()

        self.writer.grab_frame()

        if done:
            self.finish()

        return done

    def finish(self):
        if self.writer is not None:
            self.writer.finish()
            self.writer = None

    def reset(self, outfile='result.mp4'):
        self.finish()
        self.env.reset()

        Writer = manimation.writers[self.writer_type]
        self.writer = Writer(fps=self.fps)

        self.outfile = outfile
        self.writer.setup(plt.gcf(), self.outfile, self.dpi)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from azkaban.display.core import Canvas


class Table(Canvas):
    def __init__(self, groups_count, alpha_size, cmap='gist_rainbow', background=np.ones(4)):
        cm = mplcm.get_cmap(cmap)
        c_norm = colors.Normalize(vmin=0, vmax=groups_count - 1)
        self.scalar_map = mplcm.ScalarMappable(norm=c_norm, cmap=cm)
        self.background = background

        self.alphas = np.linspace(0, 1, num=alpha_size + 1)

        self.img = None
        self.reset()

    def color(self, group, alpha=-1):
        return self.scalar_map.to_rgba(group, alpha=self.alphas[alpha])

    def update(self, values):
        if self.img is None:
            plt.ion()
            self.img = plt.imshow(values)
        else:
            self.img.set_data(values)

    def reset(self):
        if self.img is not None:
            plt.ioff()
            plt.close()
        self.img = None

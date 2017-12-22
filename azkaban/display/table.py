import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from azkaban.display.core import Canvas


class Table(Canvas):
    def __init__(self, groups_count, alpha_size, group_labels=None, cmap='gist_rainbow', background=np.ones(4)):
        self.groups_count = groups_count
        self.group_labels = group_labels

        cm = mplcm.get_cmap(cmap)
        c_norm = colors.Normalize(vmin=0, vmax=self.groups_count - 1)
        self.scalar_map = mplcm.ScalarMappable(norm=c_norm, cmap=cm)
        self.background = background

        self.alphas = np.linspace(0, 1, num=alpha_size + 1)

        self.img = None
        self.reset()

    def color(self, group, alpha=-1):
        return self.scalar_map.to_rgba(group, alpha=self.alphas[alpha])

    def update(self, values):
        if self.img is None:
            # shift grid by 0.5
            height, width, _ = values.shape
            ax = plt.gca()
            ax.set_xticks(np.arange(-.5, height, 1))
            ax.set_yticks(np.arange(-.5, width, 1))
            ax.set_xticklabels(np.arange(1, height + 1, 1))
            ax.set_yticklabels(np.arange(1, width + 1, 1))

            plt.ion()
            plt.grid(True, linestyle='dotted')
            self.img = plt.imshow(values)

            if self.group_labels is not None:
                patches = []
                for idx, label in enumerate(self.group_labels):
                    patch = mpatches.Patch(color=self.color(idx), label=label)
                    patches.append(patch)
                plt.legend(handles=patches)
        else:
            self.img.set_data(values)

    def reset(self):
        if self.img is not None:
            plt.ioff()
            plt.close()
        self.img = None

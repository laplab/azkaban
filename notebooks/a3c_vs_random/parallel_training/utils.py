import numpy as np
from matplotlib import pyplot as plt
from pandas import ewma


def plot_single(name, data):
    plt.grid()
    plt.plot(data, label=name, alpha=0.2)
    plt.plot(ewma(np.array(data), span=10), label='{} ewma@10'.format(name), alpha=0.5)
    plt.plot(ewma(np.array(data), span=100), label='{} ewma@100'.format(name))
    plt.title('{} survivors'.format(name))
    plt.legend()


def plot_stats(names, stats, filename):
    data = zip(*stats)

    plt.figure(figsize=(16, 5))

    for i, (unit, name) in enumerate(zip(data, names)):
        plt.subplot(1, 2, i+1)
        plot_single(name, unit)

    plt.savefig(filename)

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np




def plot_colormap(cmap_name):
    fig, ax = plt.subplots(figsize=(6, 2))
    cmap = mpl.cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, cmap.N))
    ax.imshow([colors], extent=[0, 10, 0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(cmap_name)


if __name__ == "__main__":
    plot_colormap('rainbow')
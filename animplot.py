import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from IPython.display import HTML
sns.set()


def animate(data,fit):
    fig,axs = plt.subplots(1,1,figsize=(14,6))
    plt.close()
    def animate_func(i):
            axs.clear()
            axs.set_title(f"N={len(data[i][0])}")
            axs.plot(data[i][0],label="Hard wall")
            axs.plot(data[i][1],label="Periodic boundary")
            axs.legend()
            return axs

    anim = animation.FuncAnimation(func=animate_func,frames = len(data),fig = fig,interval =200)

    return HTML(anim.to_jshtml())



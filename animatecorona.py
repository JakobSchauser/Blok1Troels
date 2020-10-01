import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from IPython.display import HTML
sns.set()

E0 = 5
D = 3
t = 10

def solve(E0,t,N,D=0):
    H_hw = -(np.roll(np.identity(N),1)+np.roll(np.identity(N),-1))*t
    
    diag = np.ones(N)*(E0-D)
    diag[::2]+=2*D
    H_hw[np.diag_indices(N)] = diag
    
    H_pbc = H_hw.copy()
    H_pbc[[0,N-1],[N-1,0]] = -t
    
    return (np.sort(np.linalg.eig(H_hw)[0]) ,np.sort(np.linalg.eig(H_pbc)[0]) )

def animate(data):
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
    plt.show()
    return True#HTML(anim.to_jshtml())



data = [solve(E0,t,N,D=0) for N in range(6,200,8)]

animate(data)
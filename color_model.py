import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_rgb_cube():
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    # Create a grid of RGB values
    r = np.linspace(0, 1, 10)
    g = np.linspace(0, 1, 10)
    b = np.linspace(0, 1, 10)
    rg, gb, br = np.meshgrid(r, g, b)
    rg = rg.flatten()
    gb = gb.flatten()
    br = br.flatten()

    # Plot each point in the RGB color space
    for i in range(len(rg)):
        ax.scatter(rg[i], gb[i], br[i], color=(rg[i], gb[i], br[i]), s=100)

    ax.set_title('RGB Color Space')

def plot_yiq_cube():
    fig = plt.figure()
    ax = fig.add_subplot(122, projection='3d')
    ax.set_xlabel('Y')
    ax.set_ylabel('I')
    ax.set_zlabel('Q')

    # Create a grid of YIQ values
    y = np.linspace(0, 1, 10)
    i = np.linspace(-0.5957, 0.5957, 10)
    q = np.linspace(-0.5226, 0.5226, 10)
    yy, ii, qq = np.meshgrid(y, i, q)
    yy = yy.flatten()
    ii = ii.flatten()
    qq = qq.flatten()

    # Convert YIQ to RGB for plotting
    for j in range(len(yy)):
        r = yy[j] + 0.956 * ii[j] + 0.621 * qq[j]
        g = yy[j] - 0.272 * ii[j] - 0.647 * qq[j]
        b = yy[j] - 1.105 * ii[j] + 1.702 * qq[j]
        r = np.clip(r, 0, 1)
        g = np.clip(g, 0, 1)
        b = np.clip(b, 0, 1)
        ax.scatter(yy[j], ii[j], qq[j], color=(r, g, b), s=100)

    ax.set_title('YIQ Color Space')

if __name__ == "__main__":
    plot_rgb_cube()
    plot_yiq_cube()
    plt.show()
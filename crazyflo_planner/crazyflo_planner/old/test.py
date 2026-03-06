import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations, product


# visualize obstacle avoidance penalty
def plot_3d(c_obs, l_obs):
    n = 20
    offset = 0.2
    x = np.linspace(c_obs[0] - l_obs[0] - offset, c_obs[0], n)
    y = np.linspace(c_obs[1] - l_obs[1] - offset, c_obs[1], n)
    z = np.linspace(c_obs[2] - l_obs[2] - offset, c_obs[2], n)
    X, Y, Z = np.meshgrid(x, y, z)
    P = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    penalty = np.array([log_barrier(p, c_obs, l_obs) for p in P])
    penalty = penalty.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # transparent scatter with color based on penalty
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=penalty.flatten(), cmap='viridis', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plot edge of obstacle
    r = l_obs
    c = c_obs
    for s, e in combinations(np.array(list(product([c[0] - r[0], c[0] + r[0]], [c[1] - r[1], c[1] + r[1]], [c[2] - r[2], c[2] + r[2]]))), 2):
        if np.sum(np.abs(s - e) > 0.01) == 1:  # only plot edges parallel to axes
            ax.plot3D(*zip(s, e), color="red")
    # color scale
    cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Obstacle Avoidance Penalty')
    plt.title('Obstacle Avoidance Penalty')


def plot_1d_barrier(c_obs, l_obs):
    n = 100
    offset = 0.5

    fig, ax = plt.subplots()
    x = np.linspace(c_obs[0], c_obs[0] + l_obs[0] + offset, n)

    log = np.array([log_barrier(
        np.array([xi, c_obs[1], c_obs[2]]), c_obs, l_obs, d_safe=1e-3, eps=1e-6) for xi in x])
    ax.plot(x, log, label='Log1 Barrier')

    log = np.array([log_barrier(
        np.array([xi, c_obs[1], c_obs[2]]), c_obs, l_obs, d_safe=1e-6, eps=1e-6) for xi in x])
    ax.plot(x, log, label='Log2 Barrier')

    log = np.array([log_barrier(
        np.array([xi, c_obs[1], c_obs[2]]), c_obs, l_obs, d_safe=1e-9, eps=1e-6) for xi in x])
    ax.plot(x, log, label='Log3 Barrier')

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Obstacle Avoidance Penalty')
    ax.set_title('1D Slice of Obstacle Avoidance Penalty')
    ax.grid()


def log_barrier(p, c_obs, l_obs, d_safe=0.5, eps=1e-6):
    # Log barrier penalty for penetration
    gap = np.abs(p - c_obs) - l_obs  # distance to box surface
    cost = -np.log(np.prod(np.maximum(gap/d_safe, eps))) / (-np.log(eps)*3.0)  # zero at gap=d_safe, +inf at gap=0
    # if gap[0] > d_safe:
    #     cost = 0.0
    return cost


# def log_barrier(p, c_obs, l_obs, d_safe=0.5, eps=1e-6):
#     gap = ca.fabs(p - c_obs) - l_obs
#     cost = -ca.log(ca.fmax(gap, eps) / d_safe)          # zero at gap=d_safe, +inf at gap=0
#     return ca.sum1(ca.if_else(gap < d_safe, cost, 0.0))


def inverse_quadratic_barrier(p, c_obs, l_obs, d_safe=0.5):
    # CHOMP-style inverse quadratic penalty for proximity
    gap = np.abs(p - c_obs) - l_obs  # distance to box surface (negative inside)
    chomp = 0.5 * (1.0 / np.maximum(gap, 1e-6) - 1.0 / d_safe) ** 2  # quadratic penalty within d_safe, zero outside
    active = np.where(gap < d_safe, chomp, 0.0)  # only active within d_safe of surface
    return np.sum(active)


if __name__ == "__main__":
    c_obs = np.array([0.0, 0.0, 0.5])  # obstacle center
    l_obs = np.array([0.5, 0.5, 0.5])  # obstacle size (half-lengths)
    # plot_3d(c_obs, l_obs)
    plot_1d_barrier(c_obs, l_obs)
    plt.show()


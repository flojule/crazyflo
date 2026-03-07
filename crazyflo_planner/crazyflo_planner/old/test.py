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
    ax = fig.add_subplot(111, projection='3d', figsize=(12, 12))
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
    n = 1000
    offset = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    x = np.linspace(c_obs[0] - l_obs[0] - offset, c_obs[0] + l_obs[0] + offset, n)

    log = np.array([log_barrier(
        np.array([xi, c_obs[1], c_obs[2]]), c_obs, l_obs, d_safe=0.5, eps=1e-6) for xi in x])
    ax.plot(x, log, label='Log1 Barrier')

    log = np.array([log_barrier(
        np.array([xi, c_obs[1], c_obs[2]]), c_obs, l_obs, d_safe=0.2, eps=1e-6) for xi in x])
    ax.plot(x, log, label='Log2 Barrier')

    log = np.array([log_barrier(
        np.array([xi, c_obs[1], c_obs[2]]), c_obs, l_obs, d_safe=0.1, eps=1e-6) for xi in x])
    ax.plot(x, log, label='Log3 Barrier')

    # quad = np.array([quadratic_barrier(
    #     np.array([xi, c_obs[1], c_obs[2]]), c_obs, l_obs, d_safe=0.5) for xi in x])
    # ax.plot(x, quad, label='Quad1 Barrier')

    # quad = np.array([quadratic_barrier(
    #     np.array([xi, c_obs[1], c_obs[2]]), c_obs, l_obs, d_safe=0.2) for xi in x])
    # ax.plot(x, quad, label='Quad2 Barrier')

    # quad = np.array([quadratic_barrier(
    #     np.array([xi, c_obs[1], c_obs[2]]), c_obs, l_obs, d_safe=0.1) for xi in x])
    # ax.plot(x, quad, label='Quad3 Barrier')

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Obstacle Avoidance Penalty')
    ax.set_title('1D Slice of Obstacle Avoidance Penalty')
    ax.grid()
    fig.tight_layout()


def log_barrier(p, c_obs, l_obs, d_safe=0.5, eps=1e-6):
    # Log barrier penalty for penetration
    gap = (np.abs(p - c_obs) - l_obs) / d_safe  # distance to box surface, normalized
    t = np.maximum(np.minimum(gap, 1.0), 0.0)  # normalized gap with lower bound
    w = 1.0 - t**3 * (10 - 15*t + 6*t**2)
    cost = -np.log(np.maximum(gap, eps)) / (-np.log(eps))  
    # cost = -np.log(np.maximum(gap, eps)) / (-np.log(eps))
    return np.sum(w*cost) - 2.0


def quadratic_barrier(p, c_obs, l_obs, d_safe=0.5, eps=1e-1):
    # Quadratic penalty for proximity
    gap = np.abs(p - c_obs) - l_obs  # distance to box surface
    print(f"Point: {p}, Gap: {gap}")
    cost = 0.0
    if np.all(gap < 0):
        cost = 1.0
    else:
        for g in gap:
            if g < d_safe and g > 0:
                # cost += (1 - g/d_safe) ** 2  # quadratic decay to 0 at d_safe
                cost += 1 / (1 + np.exp(g-d_safe))

    return cost


# def log_barrier(p, c_obs, l_obs, d_safe=0.5, eps=1e-6):
#     # Log barrier penalty for penetration
#     gap = np.abs(p - c_obs) - l_obs  # distance to box surface
#     weight = np.abs(np.maximum(1 - (gap/d_safe)**2, 0))
#     cost = -np.log(np.prod(np.maximum(gap, eps)))
#     if np.all(gap < d_safe):
#         cost *= weight
#     else:
#         cost = 0.0
#     return np.sum(cost)


def inverse_quadratic_barrier(p, c_obs, l_obs, d_safe=0.5):
    # CHOMP-style inverse quadratic penalty for proximity
    gap = np.abs(p - c_obs) - l_obs  # distance to box surface (negative inside)
    chomp = 0.5 * (1.0 / np.maximum(gap, 1e-6) - 1.0 / d_safe) ** 2  # quadratic penalty within d_safe, zero outside
    active = np.where(gap < d_safe, chomp, 0.0)  # only active within d_safe of surface
    return np.sum(active)


if __name__ == "__main__":
    c_obs = np.array([0.0, 0.0, 0.0])  # obstacle center
    l_obs = np.array([0.5, 0.5, 0.5])  # obstacle size (half-lengths)
    # plot_3d(c_obs, l_obs)
    plot_1d_barrier(c_obs, l_obs)
    plt.show()


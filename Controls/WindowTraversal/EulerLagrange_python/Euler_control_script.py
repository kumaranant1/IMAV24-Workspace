import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def coeff_solver(i, f, T, vi, vf):
    C_hold = np.zeros((8, 3))
    X = np.array([i, f, vi, vf, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

    Tmat = np.zeros((8, 8))
    Tmat[0, 7] = 1
    Tmat[2, 6] = 1
    Tmat[4, 5] = 2
    Tmat[6, 4] = 6

    for p in range(8):
        Tmat[1, p] = T**(7 - p)

    for p in range(8):
        Tmat[3, p] = (7 - p) * T**(6 - p)

    for p in range(8):
        Tmat[5, p] = (7 - p) * (6 - p) * T**(5 - p)

    for p in range(8):
        Tmat[7, p] = (7 - p) * (6 - p) * (5 - p) * T**(4 - p)

    for q in range(3):
        C_hold[:, q] = np.linalg.solve(Tmat, X[:, q])

    return C_hold

def spline_function(i, f, W, t1, t2, v):
    Wmid = np.mean(W, axis=0)
    N = np.cross((W[1] - W[0]), (W[3] - W[0]))
    N_mag = np.linalg.norm(N)
    N_cap = N / N_mag
    k_vec = f - Wmid
    direc = np.dot(N, k_vec) / np.abs(np.dot(N, k_vec))
    n = direc * N_cap
    vel = v * n

    S_hold = np.zeros((8, 3, 2))
    S_hold[:, :, 0] = coeff_solver(i, Wmid, t1, np.array([0, 0, 0]), vel)
    S_hold[:, :, 1] = coeff_solver(Wmid, f, (t2 - t1), vel, np.array([0, 0, 0]))

    return S_hold

def visualizer(i, f, W, S, t1, t2):
    T = np.arange(0, t1, 0.01)
    pa = np.zeros((len(T), 3))

    for Tind in range(len(T)):
        ta = Tind / 100
        for u in range(3):
            for v in range(8):
                pa[Tind, u] += S[v, u, 0] * (ta**(7 - v))

    T = np.arange(0, t2 - t1, 0.01)
    pb = np.zeros((len(T), 3))

    for Tind in range(len(T)):
        tb = Tind / 100
        for u in range(3):
            for v in range(8):
                pb[Tind, u] += S[v, u, 1] * (tb**(7 - v))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    W_ = np.array([W[0], W[1], W[2], W[3], W[0]])

    ax.plot(i[0], i[1], i[2], 'gx', label='Initial Position')
    ax.plot(f[0], f[1], f[2], 'rx', label='Final Position')
    ax.plot(pa[:, 0], pa[:, 1], pa[:, 2], 'g-', label='Trajectory to Window')
    ax.plot(pb[:, 0], pb[:, 1], pb[:, 2], 'b-', label='Trajectory through Window')
    ax.plot(W_[:, 0], W_[:, 1], W_[:, 2], 'r-', label='Window Frame')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Window Traversal Simulation')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    i = np.array(eval(input('Enter initial position of drone (e.g., [x, y, z]): ')))
    f = np.array(eval(input('Enter final position of drone (e.g., [x, y, z]): ')))
    W1 = np.array(eval(input('Enter first corner of window (e.g., [x, y, z]): ')))
    W2 = np.array(eval(input('Enter second corner of window (e.g., [x, y, z]): ')))
    W3 = np.array(eval(input('Enter third corner of window (e.g., [x, y, z]): ')))
    W4 = np.array(eval(input('Enter fourth corner of window (e.g., [x, y, z]): ')))

    W = np.array([W1, W2, W3, W4])

    t1 = float(input('Enter time to reach window (in seconds): '))
    t2 = float(input('Enter time to reach final destination (in seconds): '))
    v = float(input('Enter flight speed of drone through the window: '))

    S = spline_function(i, f, W, t1, t2, v)
    visualizer(i, f, W, S, t1, t2)

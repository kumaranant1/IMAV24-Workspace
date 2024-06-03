import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def input_feasibility(b, i, f, e, Abnd):

    n = 1
    
    for B in range(4, b + 1):
        c1 = (2*(B+1)*(B-1) + B*(3*(B+1)*(B-1))**(0.5)) / ((B+1)*(B+2))
        c2 = (2*(B+1)*(B-1) - B*(3*(B+1)*(B-1))**(0.5)) / ((B+1)*(B+2))
        fa1 = abs(B*(c1**(1-(2/B)))*(f[1]-i[1])*(B-1-c1*(B+1)) / (Abnd*(1+c1)**3))
        fa2 = abs(B*(c2**(1-(2/B)))*(f[1]-i[1])*(B-1-c2*(B+1)) / (Abnd*(1+c2)**3))
        Cup = (f[0] - i[0]) * (e / (abs(i[1] - f[1]) - e))**(1/B)
        Clo = max(fa1, fa2)
        Clo = Clo**(0.5)
        
        if Clo < Cup:
            n = n + 1
    
    Sdyn = np.zeros((n,3))
    n = 0

    for B in range(4, b + 1):
        c1 = (2*(B+1)*(B-1) + B*(3*(B+1)*(B-1))**(0.5)) / ((B+1)*(B+2))
        c2 = (2*(B+1)*(B-1) - B*(3*(B+1)*(B-1))**(0.5)) / ((B+1)*(B+2))
        fa1 = abs(B*(c1**(1-(2/B)))*(f[1]-i[1])*(B-1-c1*(B+1)) / (Abnd*(1+c1)**3))
        fa2 = abs(B*(c2**(1-(2/B)))*(f[1]-i[1])*(B-1-c2*(B+1)) / (Abnd*(1+c2)**3))
        Cup = (f[0] - i[0]) * (e / (abs(i[1] - f[1]) - e))**(1/B)
        Clo = max(fa1, fa2)
        Clo = Clo**(0.5)
        
        if Clo < Cup:
            Sdyn[n, 0] = B
            Sdyn[n, 1] = Clo
            Sdyn[n, 2] = Cup
            n = n + 1
    
    return Sdyn

def window_check(i, f, W, R, Sdyni, Sdynj, Sdynk, u, v, w):
    
    c_step = 0.1
    h = 1
    g = 1
    
    for p in range(Sdyni.shape[0]):
        for Cx in np.arange(Sdyni[p, 1], Sdyni[p, 2], c_step):
            for q in range(Sdynj.shape[0]):
                for r in range(Sdynk.shape[0]):
                    x = ((W[3, u] + R - i[u]) / (f[u] - W[3, u] - R))**(1 / Sdyni[p, 0])
                    y = ((W[0, u] - R - i[u]) / (f[u] - W[0, u] + R))**(1 / Sdyni[p, 0])
                    Cylo = Cx * x * ((f[v] - W[2, v] + R) / (W[2, v] - R - i[v]))**(1 / Sdynj[q, 0])
                    Cyup = Cx * y * ((W[1, v] + R - i[v]) / (f[v] - W[1, v] - R))**(1 / Sdynj[q, 0])
                    Czlo = Cx * x * ((f[w] - W[2, w] + R) / (W[2, w] - R - i[w]))**(1 / Sdynk[r, 0])
                    Czup = Cx * y * ((W[1, w] + R - i[w]) / (f[w] - W[1, w] - R))**(1 / Sdynk[r, 0])
                    
                    if (Cylo < Cyup) and (Czlo < Czup):
                        Cylo_fes = max(Cylo, Sdynj[q, 1])
                        Cyup_fes = min(Cyup, Sdynj[q, 2])
                        Czlo_fes = max(Czlo, Sdynk[r, 1])
                        Czup_fes = min(Czup, Sdynk[r, 2])
                        if (Cylo_fes < Cyup_fes) and (Czlo_fes < Czup_fes):
                            h = h + 1

    Si_fes = np.zeros((h,8))
    h = 0

    for p in range(Sdyni.shape[0]):
        for Cx in np.arange(Sdyni[p, 1], Sdyni[p, 2], c_step):
            for q in range(Sdynj.shape[0]):
                for r in range(Sdynk.shape[0]):
                    x = ((W[3, u] + R - i[u]) / (f[u] - W[3, u] - R))**(1 / Sdyni[p, 0])
                    y = ((W[0, u] - R - i[u]) / (f[u] - W[0, u] + R))**(1 / Sdyni[p, 0])
                    Cylo = Cx * x * ((f[v] - W[2, v] + R) / (W[2, v] - R - i[v]))**(1 / Sdynj[q, 0])
                    Cyup = Cx * y * ((W[1, v] + R - i[v]) / (f[v] - W[1, v] - R))**(1 / Sdynj[q, 0])
                    Czlo = Cx * x * ((f[w] - W[2, w] + R) / (W[2, w] - R - i[w]))**(1 / Sdynk[r, 0])
                    Czup = Cx * y * ((W[1, w] + R - i[w]) / (f[w] - W[1, w] - R))**(1 / Sdynk[r, 0])
                    
                    if (Cylo < Cyup) and (Czlo < Czup):
                        Cylo_fes = max(Cylo, Sdynj[q, 1])
                        Cyup_fes = min(Cyup, Sdynj[q, 2])
                        Czlo_fes = max(Czlo, Sdynk[r, 1])
                        Czup_fes = min(Czup, Sdynk[r, 2])
                        if (Cylo_fes < Cyup_fes) and (Czlo_fes < Czup_fes):
                            Si_fes[h, 0] = Sdyni[p, 0]
                            Si_fes[h, 1] = Cx
                            Si_fes[h, 2] = Sdynj[q, 0]
                            Si_fes[h, 3] = Cylo_fes
                            Si_fes[h, 4] = Cyup_fes
                            Si_fes[h, 5] = Sdynk[r, 0]
                            Si_fes[h, 6] = Czlo_fes
                            Si_fes[h, 7] = Czup_fes
                            h = h + 1

    for m in range(Si_fes.shape[0]):
        for Cy in np.arange(Si_fes[m, 3], Si_fes[m, 4], c_step):
            for Cz in np.arange(Si_fes[m, 6], Si_fes[m, 7], c_step):
                g = g + 1
    
    Si = np.zeros((g,6))
    g = 0

    for m in range(Si_fes.shape[0]):
        for Cy in np.arange(Si_fes[m, 3], Si_fes[m, 4], c_step):
            for Cz in np.arange(Si_fes[m, 6], Si_fes[m, 7], c_step):
                Si[g, 0] = Si_fes[m, 0]
                Si[g, 1] = Si_fes[m, 1]
                Si[g, 2] = Si_fes[m, 2]
                Si[g, 3] = Cy
                Si[g, 4] = Si_fes[m, 4]
                Si[g, 5] = Cz
                g = g + 1
    
    return Si

def compile_set(i, f, W, ti, tf, e, Abnd, b, R):
    Sxdyn = input_feasibility(b, [ti, i[0]], [tf, f[0]], e, Abnd)
    Sydyn = input_feasibility(b, [ti, i[1]], [tf, f[1]], e, Abnd)
    Szdyn = input_feasibility(b, [ti, i[2]], [tf, f[2]], e, Abnd)
    
    Sx = window_check(i, f, W, R, Sxdyn, Sydyn, Szdyn, 0, 1, 2)
    Sy = window_check(i, f, W, R, Sydyn, Szdyn, Sxdyn, 1, 2, 0)
    Sz = window_check(i, f, W, R, Szdyn, Sxdyn, Sydyn, 2, 0, 1)
    
    S = np.concatenate((Sx, Sy, Sz), axis=0)
    
    return S

def create_graph(i, f, tf, S, W):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(i[0], i[1], i[2], 'x', label='Initial Position')
    ax.plot(f[0], f[1], f[2], 'X', label='Final Position')
    
    t_ = np.linspace(1, tf, num=200)
    
    for n in range(S.shape[0]):
        traj = np.zeros((len(t_), 3))
        for tidx, t in enumerate(t_):
            traj[tidx, 0] = f[0] + (i[0] - f[0]) / (1 + (t / S[n, 1])**S[n, 0])
            traj[tidx, 1] = f[1] + (i[1] - f[1]) / (1 + (t / S[n, 3])**S[n, 2])
            traj[tidx, 2] = f[2] + (i[2] - f[2]) / (1 + (t / S[n, 5])**S[n, 4])
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='g', linewidth=1)
    
    W_ = np.concatenate((W, W[0:1]), axis=0)
    ax.plot(W_[:, 0], W_[:, 1], W_[:, 2], color='r', linewidth=3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Trajectory Simulation')
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    i = np.array(eval(input('Enter initial position (e.g., [x, y, z]): ')))
    f = np.array(eval(input('Enter final position (e.g., [x, y, z]): ')))
    e = float(input('Enter tolerance for final position: '))
    tf = float(input('Enter time to reach destination: '))
    Abnd = float(input('Enter the acceleration bound of drone: '))
    b = int(input('Enter upper bound for value taken by parameter B: '))
    W1 = np.array(eval(input('Enter first corner of Window (e.g., [x, y, z]): ')))
    W2 = np.array(eval(input('Enter second corner of Window (e.g., [x, y, z]): ')))
    W3 = np.array(eval(input('Enter third corner of Window (e.g., [x, y, z]): ')))
    W4 = np.array(eval(input('Enter fourth corner of Window (e.g., [x, y, z]): ')))
    R = float(input('Enter the influence radius of drone: '))

    W = np.array([W1, W2, W3, W4])

    S = compile_set(i, f, W, 0, tf, e, Abnd, b, R)

    create_graph(i, f, tf, S, W)

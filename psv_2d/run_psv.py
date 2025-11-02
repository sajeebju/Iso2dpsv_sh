#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from propagate_psv import propagate_wave_psv

# --- domain / grid
xmax, zmax = 500.0, 300.0
nx, nz = 201, 201
dx, dz = xmax / (nx - 1), zmax / (nz - 1)

# --- source
xsrc, zsrc = 250.0, 5.0
f0  = 40.0
t0  = 4.0 / f0
tmax = 0.8

# --- model (two layers)
rho = np.ones((nx, nz), dtype=np.float32) * 1000.0
vs  = np.ones((nx, nz), dtype=np.float32) * 580.0         # upper layer Vs
vp  = np.ones((nx, nz), dtype=np.float32) * (np.sqrt(3.)*vs)  # quick Poisson-ish guess

layer_depth = 200.0
vs_low  = 480.0
vp_low  = np.sqrt(3.) * vs_low
for i in range(nx):
    for j in range(nz):
        if j*dz >= layer_depth:
            vs[i, j] = vs_low
            vp[i, j] = vp_low

# --- time step (CFL with Vp)
vmax = np.max(vp)
CC = 0.5
dt = CC * dx / vmax
nt = int(tmax / dt)

# --- absorber
w = 60
a = 0.0053

# --- accuracy order (2/4/6/8)
accuracy = 2

# --- run propagation
total_solution_space, vz = propagate_wave_psv(nx, nz, nt, dx, dz, dt, t0, f0, tmax,
                                              xsrc, zsrc, vp, vs, rho, w, a, accuracy)

# --- viz (match your SH plotting style, but for vz)
Lx, Ly = xmax, zmax
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
k = 0
mskip = 5

vscale = (vz.T - np.mean(vz))
vscale = -vscale / np.max(np.abs(vscale) + 1e-12)

def PSV_2D(_):
    global k
    wave_field = total_solution_space[:, :, k].T
    wave_field = wave_field / (np.max(np.abs(total_solution_space)) + 1e-12)
    data = wave_field + 0.0025 * vscale[:, ::-1]
    ax1.clear()
    ax1.imshow(vs.T, cmap='winter', extent=[0, Lx, Ly, 0], alpha=0.7)
    im = ax1.imshow(data, cmap='seismic', extent=[0, Lx, Ly, 0],
                    vmin=-0.08, vmax=0.08, alpha=0.6)
    ax1.set_xlim([0, Lx]); ax1.set_ylim([Ly, 0])
    ax1.set_xlabel('Distance (m)'); ax1.set_ylabel('Depth (m)')
    ax1.set_title(f'2D P–SV (vz) at {k*dt*1000:.2f} ms')
    k += mskip
    return im,

anim = animation.FuncAnimation(fig, PSV_2D, frames=int((nt - 2*mskip)/mskip),
                               interval=50, blit=True)
anim.save('PSV_wave_animation.mp4', writer='ffmpeg', fps=20)


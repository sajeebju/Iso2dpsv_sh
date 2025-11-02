#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from propagate import propagate_wave

# Setup parameters
xmax, zmax = 500.0, 300.0
nx, nz = 201, 201
Lx, Ly = xmax, zmax
dx, dz = Lx / (nx - 1), Ly / (nz - 1)
xsrc, zsrc = 250.0, 5.0

rho = np.ones((nx, nz)) * 1000.0
vs = np.ones((nx, nz)) * 580.0  # upper layer velocity

# Define layers
layer_depth = 200.0  # Depth at which the new layer starts
layer_velocity = 480.0  # Velocity of the lower layer

# Apply the second layer
for i in range(nx):
    for j in range(nz):
        if j * dz >= layer_depth:
            vs[i, j] = layer_velocity

CC = 0.5
tmax = 1.50
nt = int(tmax / (CC * dx / np.max(vs)))
dt = CC * dx / np.max(vs)
f0 = 40.0
t0 = 4.0 / 40.0
w = 60
a = 0.0053
accuracy = 2

total_solution_space, vy = propagate_wave(nx, nz, nt, dx, dz, dt, t0, f0,tmax,  xsrc, zsrc, vs, rho, w, a, accuracy)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
k = 0
mskip = 5

vscale = (vy.T - np.mean(vy))
vscale = -vscale / np.max(vscale)

def SH_2D(i):
    global k
    wave_field = total_solution_space[:, :, k].T / np.max(np.abs(total_solution_space))
    data = wave_field + 0.0025 * vscale[:, ::-1]
    ax1.clear()
    ax1.imshow(vs.T, cmap='winter', extent=[0, Lx, Ly, 0], alpha=0.7)
    im = ax1.imshow(data, cmap='seismic', extent=[0, Lx, Ly, 0], vmin=-0.08, vmax=0.08, alpha=0.6)
    ax1.set_xlim([0, Lx])
    ax1.set_ylim([Ly, 0])
    ax1.set_xlabel('Distance (m)', fontsize=12)
    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.set_title(f'2D SH at {k * dt * 1000:.2f} ms')
    k += mskip
    return im,

anim = animation.FuncAnimation(fig, SH_2D, frames=int((nt - 2 * mskip) / mskip), interval=50, blit=True)
anim.save('SH_wave_animation.mp4', writer='ffmpeg', fps=20)

#!/usr/bin/env python3
import numpy as np
from numba import jit

# --------------------------
# Helpers
# --------------------------
@jit(nopython=True)
def harm2(a, b):
    return 2.0 / (1.0/a + 1.0/b)

@jit(nopython=True)
def arith2(a, b):
    return 0.5 * (a + b)

# --------------------------
# μ directional harmonic averages (reuse your SH pattern)
# --------------------------
@jit(nopython=True)
def shear_avg_2nd_order(mu, nx, nz, mux, muz):
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            mux[i,j] = 2.0 / (1.0/mu[i+1,j] + 1.0/mu[i,j])
            muz[i,j] = 2.0 / (1.0/mu[i,j+1] + 1.0/mu[i,j])
    return mux, muz

@jit(nopython=True)
def shear_avg_4th_order(mu, nx, nz, mux, muz):
    for i in range(2, nx - 2):
        for j in range(2, nz - 2):
            mux[i,j] = 2.0 / (1.0/mu[i+2,j] + 1.0/mu[i+1,j] + 1.0/mu[i,j])
            muz[i,j] = 2.0 / (1.0/mu[i,j+2] + 1.0/mu[i,j+1] + 1.0/mu[i,j])
    return mux, muz

@jit(nopython=True)
def shear_avg_6th_order(mu, nx, nz, mux, muz):
    for i in range(3, nx - 3):
        for j in range(3, nz - 3):
            mux[i,j] = 2.0 / (1.0/mu[i+3,j] + 1.0/mu[i+2,j] + 1.0/mu[i+1,j] + 1.0/mu[i,j])
            muz[i,j] = 2.0 / (1.0/mu[i,j+3] + 1.0/mu[i,j+2] + 1.0/mu[i,j+1] + 1.0/mu[i,j])
    return mux, muz

@jit(nopython=True)
def shear_avg_8th_order(mu, nx, nz, mux, muz):
    for i in range(4, nx - 4):
        for j in range(4, nz - 4):
            mux[i,j] = 2.0 / (1.0/mu[i+4,j] + 1.0/mu[i+3,j] + 1.0/mu[i+2,j] + 1.0/mu[i+1,j] + 1.0/mu[i,j])
            muz[i,j] = 2.0 / (1.0/mu[i,j+4] + 1.0/mu[i,j+3] + 1.0/mu[i,j+2] + 1.0/mu[i,j+1] + 1.0/mu[i,j])
    return mux, muz

# --------------------------
# λ face averages (arithmetic)
# --------------------------
@jit(nopython=True)
def lam_avg_2nd_order(lam, nx, nz, lamx, lamz):
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            lamx[i,j] = arith2(lam[i+1,j], lam[i,j])
            lamz[i,j] = arith2(lam[i,j+1], lam[i,j])
    return lamx, lamz

@jit(nopython=True)
def lam_avg_4th_order(lam, nx, nz, lamx, lamz):
    for i in range(2, nx - 2):
        for j in range(2, nz - 2):
            lamx[i,j] = (lam[i+2,j] + lam[i+1,j] + lam[i,j]) / 3.0
            lamz[i,j] = (lam[i,j+2] + lam[i,j+1] + lam[i,j]) / 3.0
    return lamx, lamz

@jit(nopython=True)
def lam_avg_6th_order(lam, nx, nz, lamx, lamz):
    for i in range(3, nx - 3):
        for j in range(3, nz - 3):
            lamx[i,j] = (lam[i+3,j] + lam[i+2,j] + lam[i+1,j] + lam[i,j]) / 4.0
            lamz[i,j] = (lam[i,j+3] + lam[i,j+2] + lam[i,j+1] + lam[i,j]) / 4.0
    return lamx, lamz

@jit(nopython=True)
def lam_avg_8th_order(lam, nx, nz, lamx, lamz):
    for i in range(4, nx - 4):
        for j in range(4, nz - 4):
            lamx[i,j] = (lam[i+4,j] + lam[i+3,j] + lam[i+2,j] + lam[i+1,j] + lam[i,j]) / 5.0
            lamz[i,j] = (lam[i,j+4] + lam[i,j+3] + lam[i,j+2] + lam[i,j+1] + lam[i,j]) / 5.0
    return lamx, lamz

# --------------------------
# (λ+2μ) face averages (arithmetic)
# --------------------------
@jit(nopython=True)
def a2_avg_2nd_order(lam, mu, nx, nz, a2x, a2z):
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            a2x[i,j] = arith2(lam[i+1,j] + 2.0*mu[i+1,j], lam[i,j] + 2.0*mu[i,j])
            a2z[i,j] = arith2(lam[i,j+1] + 2.0*mu[i,j+1], lam[i,j] + 2.0*mu[i,j])
    return a2x, a2z

@jit(nopython=True)
def a2_avg_4th_order(lam, mu, nx, nz, a2x, a2z):
    for i in range(2, nx - 2):
        for j in range(2, nz - 2):
            a2x[i,j] = ((lam[i+2,j]+2*mu[i+2,j]) + (lam[i+1,j]+2*mu[i+1,j]) + (lam[i,j]+2*mu[i,j])) / 3.0
            a2z[i,j] = ((lam[i,j+2]+2*mu[i,j+2]) + (lam[i,j+1]+2*mu[i,j+1]) + (lam[i,j]+2*mu[i,j])) / 3.0
    return a2x, a2z

@jit(nopython=True)
def a2_avg_6th_order(lam, mu, nx, nz, a2x, a2z):
    for i in range(3, nx - 3):
        for j in range(3, nz - 3):
            a2x[i,j] = ((lam[i+3,j]+2*mu[i+3,j]) + (lam[i+2,j]+2*mu[i+2,j]) +
                        (lam[i+1,j]+2*mu[i+1,j]) + (lam[i,j]+2*mu[i,j])) / 4.0
            a2z[i,j] = ((lam[i,j+3]+2*mu[i,j+3]) + (lam[i,j+2]+2*mu[i,j+2]) +
                        (lam[i,j+1]+2*mu[i,j+1]) + (lam[i,j]+2*mu[i,j])) / 4.0
    return a2x, a2z

@jit(nopython=True)
def a2_avg_8th_order(lam, mu, nx, nz, a2x, a2z):
    for i in range(4, nx - 4):
        for j in range(4, nz - 4):
            a2x[i,j] = ((lam[i+4,j]+2*mu[i+4,j]) + (lam[i+3,j]+2*mu[i+3,j]) +
                        (lam[i+2,j]+2*mu[i+2,j]) + (lam[i+1,j]+2*mu[i+1,j]) +
                        (lam[i,j]+2*mu[i,j])) / 5.0
            a2z[i,j] = ((lam[i,j+4]+2*mu[i,j+4]) + (lam[i,j+3]+2*mu[i,j+3]) +
                        (lam[i,j+2]+2*mu[i,j+2]) + (lam[i,j+1]+2*mu[i,j+1]) +
                        (lam[i,j]+2*mu[i,j])) / 5.0
    return a2x, a2z

# --------------------------
# Velocity updates
# --------------------------
@jit(nopython=True)
def update_vel_2nd_order(vx, vz, sxx, szz, sxz, dx, dz, dt, nx, nz, rho):
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            sxx_x = (sxx[i, j] - sxx[i-1, j]) / dx
            sxz_z = (sxz[i, j] - sxz[i, j-1]) / dz
            sxz_x = (sxz[i, j] - sxz[i-1, j]) / dx
            szz_z = (szz[i, j] - szz[i, j-1]) / dz
            invr = 1.0 / rho[i, j]
            vx[i, j] += dt * invr * (sxx_x + sxz_z)
            vz[i, j] += dt * invr * (sxz_x + szz_z)
    return vx, vz

@jit(nopython=True)
def update_vel_4th_order(vx, vz, sxx, szz, sxz, dx, dz, dt, nx, nz, rho):
    for i in range(2, nx - 2):
        for j in range(2, nz - 2):
            sxx_x = (-sxx[i-2,j] + 8*sxx[i-1,j] - 8*sxx[i+1,j] + sxx[i+2,j]) / (12*dx)
            sxz_z = (-sxz[i,j-2] + 8*sxz[i,j-1] - 8*sxz[i,j+1] + sxz[i,j+2]) / (12*dz)
            sxz_x = (-sxz[i-2,j] + 8*sxz[i-1,j] - 8*sxz[i+1,j] + sxz[i+2,j]) / (12*dx)
            szz_z = (-szz[i,j-2] + 8*szz[i,j-1] - 8*szz[i,j+1] + szz[i,j+2]) / (12*dz)
            invr = 1.0 / rho[i, j]
            vx[i, j] += dt * invr * (sxx_x + sxz_z)
            vz[i, j] += dt * invr * (sxz_x + szz_z)
    return vx, vz

@jit(nopython=True)
def update_vel_6th_order(vx, vz, sxx, szz, sxz, dx, dz, dt, nx, nz, rho):
    for i in range(3, nx - 3):
        for j in range(3, nz - 3):
            sxx_x = (-sxx[i-3,j] + 9*sxx[i-2,j] - 45*sxx[i-1,j] + 45*sxx[i+1,j] - 9*sxx[i+2,j] + sxx[i+3,j]) / (60*dx)
            sxz_z = (-sxz[i,j-3] + 9*sxz[i,j-2] - 45*sxz[i,j-1] + 45*sxz[i,j+1] - 9*sxz[i,j+2] + sxz[i,j+3]) / (60*dz)
            sxz_x = (-sxz[i-3,j] + 9*sxz[i-2,j] - 45*sxz[i-1,j] + 45*sxz[i+1,j] - 9*sxz[i+2,j] + sxz[i+3,j]) / (60*dx)
            szz_z = (-szz[i,j-3] + 9*szz[i,j-2] - 45*szz[i,j-1] + 45*szz[i,j+1] - 9*szz[i,j+2] + szz[i,j+3]) / (60*dz)
            invr = 1.0 / rho[i, j]
            vx[i, j] += dt * invr * (sxx_x + sxz_z)
            vz[i, j] += dt * invr * (sxz_x + szz_z)
    return vx, vz

@jit(nopython=True)
def update_vel_8th_order(vx, vz, sxx, szz, sxz, dx, dz, dt, nx, nz, rho):
    for i in range(4, nx - 4):
        for j in range(4, nz - 4):
            sxx_x = (-sxx[i-4,j] + 8*sxx[i-3,j] - 28*sxx[i-2,j] + 56*sxx[i-1,j]
                     - 56*sxx[i+1,j] + 28*sxx[i+2,j] - 8*sxx[i+3,j] + sxx[i+4,j]) / (280*dx)
            sxz_z = (-sxz[i,j-4] + 8*sxz[i,j-3] - 28*sxz[i,j-2] + 56*sxz[i,j-1]
                     - 56*sxz[i,j+1] + 28*sxz[i,j+2] - 8*sxz[i,j+3] + sxz[i,j+4]) / (280*dz)
            sxz_x = (-sxz[i-4,j] + 8*sxz[i-3,j] - 28*sxz[i-2,j] + 56*sxz[i-1,j]
                     - 56*sxz[i+1,j] + 28*sxz[i+2,j] - 8*sxz[i+3,j] + sxz[i+4,j]) / (280*dx)
            szz_z = (-szz[i,j-4] + 8*szz[i,j-3] - 28*szz[i,j-2] + 56*szz[i,j-1]
                     - 56*szz[i,j+1] + 28*szz[i,j+2] - 8*szz[i,j+3] + szz[i,j+4]) / (280*dz)
            invr = 1.0 / rho[i, j]
            vx[i, j] += dt * invr * (sxx_x + sxz_z)
            vz[i, j] += dt * invr * (sxz_x + szz_z)
    return vx, vz

# --------------------------
# Stress updates (with directional averages)
# --------------------------
@jit(nopython=True)
def update_stress_2nd_order(vx, vz, sxx, szz, sxz,
                            dx, dz, dt, nx, nz,
                            lamx, lamz, a2x, a2z, mux, muz):
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            vx_x = (vx[i+1, j] - vx[i, j]) / dx
            vz_z = (vz[i, j+1] - vz[i, j]) / dz
            vz_x = (vz[i+1, j] - vz[i, j]) / dx
            vx_z = (vx[i, j+1] - vx[i, j]) / dz
            sxx[i, j] += dt * ( a2x[i, j] * vx_x + lamz[i, j] * vz_z )
            szz[i, j] += dt * ( lamx[i, j] * vx_x + a2z[i, j] * vz_z )
            sxz[i, j] += dt * ( muz[i, j]  * vx_z + mux[i, j]  * vz_x )
    return sxx, szz, sxz

@jit(nopython=True)
def update_stress_4th_order(vx, vz, sxx, szz, sxz,
                            dx, dz, dt, nx, nz,
                            lamx, lamz, a2x, a2z, mux, muz):
    for i in range(2, nx - 2):
        for j in range(2, nz - 2):
            vx_x = (-vx[i-2,j] + 8*vx[i-1,j] - 8*vx[i+1,j] + vx[i+2,j]) / (12*dx)
            vz_z = (-vz[i,j-2] + 8*vz[i,j-1] - 8*vz[i,j+1] + vz[i,j+2]) / (12*dz)
            vz_x = (-vz[i-2,j] + 8*vz[i-1,j] - 8*vz[i+1,j] + vz[i+2,j]) / (12*dx)
            vx_z = (-vx[i,j-2] + 8*vx[i,j-1] - 8*vx[i,j+1] + vx[i,j+2]) / (12*dz)
            sxx[i, j] += dt * ( a2x[i, j] * vx_x + lamz[i, j] * vz_z )
            szz[i, j] += dt * ( lamx[i, j] * vx_x + a2z[i, j] * vz_z )
            sxz[i, j] += dt * ( muz[i, j]  * vx_z + mux[i, j]  * vz_x )
    return sxx, szz, sxz

@jit(nopython=True)
def update_stress_6th_order(vx, vz, sxx, szz, sxz,
                            dx, dz, dt, nx, nz,
                            lamx, lamz, a2x, a2z, mux, muz):
    for i in range(3, nx - 3):
        for j in range(3, nz - 3):
            vx_x = (-vx[i-3,j] + 9*vx[i-2,j] - 45*vx[i-1,j] + 45*vx[i+1,j] - 9*vx[i+2,j] + vx[i+3,j]) / (60*dx)
            vz_z = (-vz[i,j-3] + 9*vz[i,j-2] - 45*vz[i,j-1] + 45*vz[i,j+1] - 9*vz[i,j+2] + vz[i,j+3]) / (60*dz)
            vz_x = (-vz[i-3,j] + 9*vz[i-2,j] - 45*vz[i-1,j] + 45*vz[i+1,j] - 9*vz[i+2,j] + vz[i+3,j]) / (60*dx)
            vx_z = (-vx[i,j-3] + 9*vx[i,j-2] - 45*vx[i,j-1] + 45*vx[i,j+1] - 9*vx[i,j+2] + vx[i,j+3]) / (60*dz)
            sxx[i, j] += dt * ( a2x[i, j] * vx_x + lamz[i, j] * vz_z )
            szz[i, j] += dt * ( lamx[i, j] * vx_x + a2z[i, j] * vz_z )
            sxz[i, j] += dt * ( muz[i, j]  * vx_z + mux[i, j]  * vz_x )
    return sxx, szz, sxz

@jit(nopython=True)
def update_stress_8th_order(vx, vz, sxx, szz, sxz,
                            dx, dz, dt, nx, nz,
                            lamx, lamz, a2x, a2z, mux, muz):
    for i in range(4, nx - 4):
        for j in range(4, nz - 4):
            vx_x = (-vx[i-4,j] + 8*vx[i-3,j] - 28*vx[i-2,j] + 56*vx[i-1,j]
                    - 56*vx[i+1,j] + 28*vx[i+2,j] - 8*vx[i+3,j] + vx[i+4,j]) / (280*dx)
            vz_z = (-vz[i,j-4] + 8*vz[i,j-3] - 28*vz[i,j-2] + 56*vz[i,j-1]
                    - 56*vz[i,j+1] + 28*vz[i,j+2] - 8*vz[i,j+3] + vz[i,j+4]) / (280*dz)
            vz_x = (-vz[i-4,j] + 8*vz[i-3,j] - 28*vz[i-2,j] + 56*vz[i-1,j]
                    - 56*vz[i+1,j] + 28*vz[i+2,j] - 8*vz[i+3,j] + vz[i+4,j]) / (280*dx)
            vx_z = (-vx[i,j-4] + 8*vx[i,j-3] - 28*vx[i,j-2] + 56*vx[i,j-1]
                    - 56*vx[i,j+1] + 28*vx[i,j+2] - 8*vx[i,j+3] + vx[i,j+4]) / (280*dz)
            sxx[i, j] += dt * ( a2x[i, j] * vx_x + lamz[i, j] * vz_z )
            szz[i, j] += dt * ( lamx[i, j] * vx_x + a2z[i, j] * vz_z )
            sxz[i, j] += dt * ( muz[i, j]  * vx_z + mux[i, j]  * vz_x )
    return sxx, szz, sxz


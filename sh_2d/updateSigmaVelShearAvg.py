#!/usr/bin/env python3
import numpy as np
from numba import jit

@jit(nopython=True)
def update_vel_2nd_order(vy, syx, syz, dx, dz, dt, nx, nz, rho):
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            syx_x = (syx[i,j] - syx[i - 1,j]) / dx
            syz_z = (syz[i,j] - syz[i,j - 1]) / dz
            vy[i,j] = vy[i,j] + (dt/rho[i,j]) * (syx_x + syz_z)
    return vy


@jit(nopython=True)
def update_vel_4th_order(vy, syx, syz, dx, dz, dt, nx, nz, rho):
    for i in range(2, nx - 2): 
        for j in range(2, nz - 2):
            syx_x = (-syx[i-2,j] + 8*syx[i-1,j] - 8*syx[i+1,j] + syx[i+2,j]) / (12 * dx)
            syz_z = (-syz[i,j-2] + 8*syz[i,j-1] - 8*syz[i,j+1] + syz[i,j+2]) / (12 * dz)
            vy[i,j] += (dt / rho[i,j]) * (syx_x + syz_z)

    return vy


@jit(nopython=True) 
def update_vel_6th_order(vy, syx, syz, dx, dz, dt, nx, nz, rho):
    # 6th-order FD operator
    for i in range(3, nx - 3):  
        for j in range(3, nz - 3):
            syx_x = (-syx[i-3, j] + 9*syx[i-2, j] - 45*syx[i-1, j] +
                     45*syx[i+1, j] - 9*syx[i+2, j] + syx[i+3, j]) / (60 * dx)
            syz_z = (-syz[i, j-3] + 9*syz[i, j-2] - 45*syz[i, j-1] +
                     45*syz[i, j+1] - 9*syz[i, j+2] + syz[i, j+3]) / (60 * dz)
            vy[i, j] += (dt / rho[i, j]) * (syx_x + syz_z)

    return vy


@jit(nopython=True) 
def update_vel_8th_order(vy, syx, syz, dx, dz, dt, nx, nz, rho):
    for i in range(4, nx - 4): 
        for j in range(4, nz - 4):
            syx_x = (-syx[i-4, j] + 8*syx[i-3, j] - 28*syx[i-2, j] +
                     56*syx[i-1, j] - 56*syx[i+1, j] + 28*syx[i+2, j] -
                     8*syx[i+3, j] + syx[i+4, j]) / (280 * dx)
            syz_z = (-syz[i, j-4] + 8*syz[i, j-3] - 28*syz[i, j-2] +
                     56*syz[i, j-1] - 56*syz[i, j+1] + 28*syz[i, j+2] -
                     8*syz[i, j+3] + syz[i, j+4]) / (280 * dz)
            vy[i, j] += (dt / rho[i, j]) * (syx_x + syz_z)

    return vy


@jit(nopython=True) 
def update_stress_2nd_order(vy, syx, syz, dx, dz, dt, nx, nz, mux, muz):
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            vy_x = (vy[i + 1,j] - vy[i,j]) / dx
            vy_z = (vy[i,j + 1] - vy[i,j]) / dz
            syx[i,j] = syx[i,j] + dt * mux[i,j] * vy_x
            syz[i,j] = syz[i,j] + dt * muz[i,j] * vy_z

    return syx, syz

@jit(nopython=True)
def update_stress_4th_order(vy, syx, syz, dx, dz, dt, nx, nz, mux, muz):
    for i in range(2, nx - 2):
        for j in range(2, nz - 2):
            vy_x = (-vy[i-2,j] + 8*vy[i-1,j] - 8*vy[i+1,j] + vy[i+2,j]) / (12 * dx)
            vy_z = (-vy[i,j-2] + 8*vy[i,j-1] - 8*vy[i,j+1] + vy[i,j+2]) / (12 * dz)
            syx[i,j] += dt * mux[i,j] * vy_x
            syz[i,j] += dt * muz[i,j] * vy_z

    return syx, syz


@jit(nopython=True)
def update_stress_6th_order(vy, syx, syz, dx, dz, dt, nx, nz, mux, muz):
    for i in range(3, nx - 3):
        for j in range(3, nz - 3):
            vy_x = (-vy[i-3, j] + 9*vy[i-2, j] - 45*vy[i-1, j] + 45*vy[i+1, j] - 9*vy[i+2, j] + vy[i+3, j]) / (60 * dx)
            vy_z = (-vy[i, j-3] + 9*vy[i, j-2] - 45*vy[i, j-1] + 45*vy[i, j+1] - 9*vy[i, j+2] + vy[i, j+3]) / (60 * dz)
            syx[i, j] += dt * mux[i, j] * vy_x
            syz[i, j] += dt * muz[i, j] * vy_z
    return syx, syz

@jit(nopython=True)
def update_stress_8th_order(vy, syx, syz, dx, dz, dt, nx, nz, mux, muz):
    for i in range(4, nx - 4):
        for j in range(4, nz - 4):
            vy_x = (-vy[i-4, j] + 8*vy[i-3, j] - 28*vy[i-2, j] + 56*vy[i-1, j] - 56*vy[i+1, j] + 28*vy[i+2, j] - 8*vy[i+3, j] + vy[i+4, j]) / (280 * dx)
            vy_z = (-vy[i, j-4] + 8*vy[i, j-3] - 28*vy[i, j-2] + 56*vy[i, j-1] - 56*vy[i, j+1] + 28*vy[i, j+2] - 8*vy[i, j+3] + vy[i, j+4]) / (280 * dz)
            syx[i, j] += dt * mux[i, j] * vy_x
            syz[i, j] += dt * muz[i, j] * vy_z
    return syx, syz

@jit(nopython=True)
def shear_avg_2nd_order(mu, nx, nz, mux, muz):
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            mux[i,j] = 2 / (1 / mu[i + 1,j] + 1 / mu[i,j])
            muz[i,j] = 2 / (1 / mu[i,j + 1] + 1 / mu[i,j])
    return mux, muz

@jit(nopython=True)
def shear_avg_4th_order(mu, nx, nz, mux, muz):
    for i in range(2, nx - 2):
        for j in range(2, nz - 2):
            mux[i, j] = 2 / (1 / mu[i+2, j] + 1 / mu[i+1, j] + 1 / mu[i, j])
            muz[i, j] = 2 / (1 / mu[i, j+2] + 1 / mu[i, j+1] + 1 / mu[i, j])
    return mux, muz

@jit(nopython=True)
def shear_avg_6th_order(mu, nx, nz, mux, muz):
    for i in range(3, nx - 3):
        for j in range(3, nz - 3):
            mux[i, j] = 2 / (1 / mu[i+3, j] + 1 / mu[i+2, j] + 1 / mu[i+1, j] + 1 / mu[i, j])
            muz[i, j] = 2 / (1 / mu[i, j+3] + 1 / mu[i, j+2] + 1 / mu[i, j+1] + 1 / mu[i, j])
    return mux, muz

@jit(nopython=True)
def shear_avg_8th_order(mu, nx, nz, mux, muz):
    for i in range(4, nx - 4):
        for j in range(4, nz - 4):
            mux[i, j] = 2 / (1 / mu[i+4, j] + 1 / mu[i+3, j] + 1 / mu[i+2, j] + 1 / mu[i+1, j] + 1 / mu[i, j])
            muz[i, j] = 2 / (1 / mu[i, j+4] + 1 / mu[i, j+3] + 1 / mu[i, j+2] + 1 / mu[i, j+1] + 1 / mu[i, j])
    return mux, muz


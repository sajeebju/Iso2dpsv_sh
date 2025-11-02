#!/usr/bin/env python3
import numpy as np
from absorber import Absorber
from updateSigmaVelPSV import *

def propagate_wave_psv(nx, nz, nt, dx, dz, dt, t0, f0, tmax,
                       xsrc, zsrc, vp, vs, rho,
                       w, a, accuracy):
    """
    Collocated-grid 2D P–SV propagation (vx,vz,sxx,szz,sxz), using your absorber.

    Source: isotropic pressure pulse injected equally into sxx & szz:
        ST(t) = -2 (t - t0) f0^2 exp(-(f0^2)(t - t0)^2)
    """

    # Allocate fields
    vx  = np.zeros((nx, nz), dtype=np.float32)
    vz  = np.zeros((nx, nz), dtype=np.float32)
    sxx = np.zeros((nx, nz), dtype=np.float32)
    szz = np.zeros((nx, nz), dtype=np.float32)
    sxz = np.zeros((nx, nz), dtype=np.float32)

    # Material
    lam = rho*(vp*vp - 2.0*vs*vs)   # λ
    mu  = rho*(vs*vs)               # μ

    # Directional averages (face-centered)
    lamx = np.zeros_like(lam); lamz = np.zeros_like(lam)
    a2x  = np.zeros_like(lam); a2z  = np.zeros_like(lam)   # a2 = λ + 2μ
    mux  = np.zeros_like(mu);  muz  = np.zeros_like(mu)

    # Build directional averages according to accuracy
    if accuracy == 2:
        mux, muz   = shear_avg_2nd_order(mu, nx, nz, mux, muz)
        lamx, lamz = lam_avg_2nd_order(lam, nx, nz, lamx, lamz)
        a2x, a2z   = a2_avg_2nd_order(lam, mu, nx, nz, a2x, a2z)
        upd_vel    = update_vel_2nd_order
        upd_stress = update_stress_2nd_order
    elif accuracy == 4:
        mux, muz   = shear_avg_4th_order(mu, nx, nz, mux, muz)
        lamx, lamz = lam_avg_4th_order(lam, nx, nz, lamx, lamz)
        a2x, a2z   = a2_avg_4th_order(lam, mu, nx, nz, a2x, a2z)
        upd_vel    = update_vel_4th_order
        upd_stress = update_stress_4th_order
    elif accuracy == 6:
        mux, muz   = shear_avg_6th_order(mu, nx, nz, mux, muz)
        lamx, lamz = lam_avg_6th_order(lam, nx, nz, lamx, lamz)
        a2x, a2z   = a2_avg_6th_order(lam, mu, nx, nz, a2x, a2z)
        upd_vel    = update_vel_6th_order
        upd_stress = update_stress_6th_order
    elif accuracy == 8:
        mux, muz   = shear_avg_8th_order(mu, nx, nz, mux, muz)
        lamx, lamz = lam_avg_8th_order(lam, nx, nz, lamx, lamz)
        a2x, a2z   = a2_avg_8th_order(lam, mu, nx, nz, a2x, a2z)
        upd_vel    = update_vel_8th_order
        upd_stress = update_stress_8th_order
    else:
        raise ValueError("accuracy must be 2, 4, 6, or 8")

    # Source (same wavelet style as your SH code)
    isx, isz = int(xsrc / dx), int(zsrc / dz)
    t = np.linspace(0.0, tmax, nt)
    ST = -2.0 * (t - t0) * (f0**2) * np.exp(- (f0**2) * (t - t0)**2)

    # Absorber
    my_absorber  = Absorber(nx, nz, w, a)
    absorb_coeff = my_absorber.compute_absorption_coefficients()

    # Output wavefield (save vz like you saved vy)
    solution_space = np.zeros((nx, nz, nt), dtype=np.float32)

    for it in range(nt):
        # --- Stress update
        sxx, szz, sxz = upd_stress(vx, vz, sxx, szz, sxz,
                                   dx, dz, dt, nx, nz,
                                   lamx, lamz, a2x, a2z, mux, muz)

        # Inject isotropic pressure source (adds equally to normal stresses)
        sxx[isx, isz] += dt * ST[it]
        szz[isx, isz] += dt * ST[it]

        # --- Velocity update
        vx, vz = upd_vel(vx, vz, sxx, szz, sxz,
                         dx, dz, dt, nx, nz, rho)

        # --- Absorbing sponge (your style: multiply fields)
        vx  *= absorb_coeff
        vz  *= absorb_coeff
        sxx *= absorb_coeff
        szz *= absorb_coeff
        sxz *= absorb_coeff

        solution_space[:, :, it] = vz

    return solution_space, vz

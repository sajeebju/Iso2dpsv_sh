#!/usr/bin/env python3
import numpy as np
from updateSigmaVelPSV import *
import matplotlib.pyplot as plt

def PSV_SEIS(nx, nz, nt, dx, dz, dt, f0, t0, isrc, jsrc, ir, jr, vp, vs, rho, accuracy):
    """
    Compute a synthetic seismogram (vertical component vz) for 2D P–SV propagation
    using the collocated velocity-stress finite-difference scheme.

    Returns
    -------
    time : np.ndarray
        Time vector [s]
    seis : np.ndarray
        Vertical velocity seismogram at receiver [vz(ir, jr, t)]
    """

    if accuracy not in [2, 4, 6, 8]:
        raise ValueError("Unsupported accuracy level. Choose among 2, 4, 6, or 8.")

    # -------------------------------------------
    # Source time function (1st derivative Gaussian)
    # -------------------------------------------
    time = np.linspace(0, nt * dt, nt)
    src = -2.0 * (time - t0) * (f0 ** 2) * np.exp(- (f0 ** 2) * (time - t0) ** 2)

    # -------------------------------------------
    # Initialize fields
    # -------------------------------------------
    vx  = np.zeros((nx, nz))   # particle velocity x
    vz  = np.zeros((nx, nz))   # particle velocity z
    sxx = np.zeros((nx, nz))   # normal stress σxx
    szz = np.zeros((nx, nz))   # normal stress σzz
    sxz = np.zeros((nx, nz))   # shear stress σxz

    # -------------------------------------------
    # Elastic parameters
    # -------------------------------------------
    lam = rho * (vp ** 2 - 2.0 * vs ** 2)
    mu  = rho * (vs ** 2)

    # Averages for staggered updates
    lamx = np.zeros_like(lam); lamz = np.zeros_like(lam)
    a2x  = np.zeros_like(lam); a2z  = np.zeros_like(lam)
    mux  = np.zeros_like(mu);  muz  = np.zeros_like(mu)

    # -------------------------------------------
    # Initialize harmonic averages based on accuracy
    # -------------------------------------------
    if accuracy == 2:
        mux, muz   = shear_avg_2nd_order(mu, nx, nz, mux, muz)
        lamx, lamz = lam_avg_2nd_order(lam, nx, nz, lamx, lamz)
        a2x, a2z   = a2_avg_2nd_order(lam, mu, nx, nz, a2x, a2z)
        upd_vel, upd_stress = update_vel_2nd_order, update_stress_2nd_order
    elif accuracy == 4:
        mux, muz   = shear_avg_4th_order(mu, nx, nz, mux, muz)
        lamx, lamz = lam_avg_4th_order(lam, nx, nz, lamx, lamz)
        a2x, a2z   = a2_avg_4th_order(lam, mu, nx, nz, a2x, a2z)
        upd_vel, upd_stress = update_vel_4th_order, update_stress_4th_order
    elif accuracy == 6:
        mux, muz   = shear_avg_6th_order(mu, nx, nz, mux, muz)
        lamx, lamz = lam_avg_6th_order(lam, nx, nz, lamx, lamz)
        a2x, a2z   = a2_avg_6th_order(lam, mu, nx, nz, a2x, a2z)
        upd_vel, upd_stress = update_vel_6th_order, update_stress_6th_order
    elif accuracy == 8:
        mux, muz   = shear_avg_8th_order(mu, nx, nz, mux, muz)
        lamx, lamz = lam_avg_8th_order(lam, nx, nz, lamx, lamz)
        a2x, a2z   = a2_avg_8th_order(lam, mu, nx, nz, a2x, a2z)
        upd_vel, upd_stress = update_vel_8th_order, update_stress_8th_order

    # -------------------------------------------
    # Seismogram storage
    # -------------------------------------------
    seis = np.zeros(nt)
    solution_space = np.zeros((nx, nz, nt))

    # -------------------------------------------
    # Finite-difference time stepping
    # -------------------------------------------

    for it in range(nt):
        # --- inject volumetric (P-type) source ---
        Ksrc = lam[isrc, jsrc] + 2.0/3.0 * mu[isrc, jsrc]
        amp  = 1e-6 * Ksrc
        sxx[isrc, jsrc] += amp * src[it]
        szz[isrc, jsrc] += amp * src[it]
    
        # --- update stresses ---
        sxx, szz, sxz = upd_stress(
            vx, vz, sxx, szz, sxz,
            dx, dz, dt, nx, nz,
            lamx, lamz, a2x, a2z, mux, muz
        )
    
        # --- update velocities ---
        vx, vz = upd_vel(vx, vz, sxx, szz, sxz,
                         dx, dz, dt, nx, nz, rho)
    
        seis[it] = vz[ir, jr]
        
        solution_space[:, :, it] = vz

    return time, seis




# --- Model parameters ---
dx, dz, dt = 1.0, 1.0, 0.001
xmax, zmax, tmax = 500.0, 500.0, 0.502
nx, nz, nt = int(xmax/dx), int(zmax/dz), int(tmax/dt)

vs0 = 580.0
vp0 = np.sqrt(3.0) * vs0
rho0 = 1000.0

vs = np.ones((nx, nz)) * vs0
vp = np.ones((nx, nz)) * vp0
rho = np.ones((nx, nz)) * rho0

xsrc, zsrc = 250.0, 250.0
xr, zr = 330.0, 330.0
isrc, jsrc = int(xsrc/dx), int(zsrc/dz)
ir, jr = int(xr/dx), int(zr/dz)

f0 = 40.0
t0 = 4.0 / f0
accuracy = 2

# --- Run seismogram ---
time, seis = PSV_SEIS(nx, nz, nt, dx, dz, dt,
                      f0, t0, isrc, jsrc, ir, jr,
                      vp, vs, rho, accuracy)

# --- Plot ---
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(time, seis, color='blue', lw=1.4)
plt.xlabel("Time (s)")
plt.ylabel("Vertical velocity (vz)")
plt.title("Synthetic Seismogram (P–SV Wavefield)")
plt.grid(True)
plt.tight_layout()
plt.show()



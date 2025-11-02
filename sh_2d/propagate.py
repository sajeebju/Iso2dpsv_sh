#!/usr/bin/env python3
import numpy as np
from absorber import Absorber
from updateSigmaVelShearAvg import *

def propagate_wave(nx, nz, nt, dx, dz, dt, t0, f0,tmax, xsrc, zsrc, vs, rho, w, a, accuracy):

  vy = np.zeros((nx, nz))         
  syx = np.zeros((nx, nz))    
  syz = np.zeros((nx, nz))      
  mux = np.zeros((nx, nz))    
  muz = np.zeros((nx, nz))     
 
  mu = np.zeros((nx, nz))
  mu = rho * vs**2 
  
  mux = mu
  muz = mu

  isx, isz = int(xsrc / dx), int(zsrc / dz)  
  t = np.linspace(0, tmax, nt) 
  ST = -2. * (t - t0) * (f0 ** 2) * (np.exp(- (f0 ** 2) * (t - t0) ** 2))

  if accuracy == 2:
    mux, muz = shear_avg_2nd_order(mu, nx, nz, mux, muz)
  elif accuracy == 4:
      mux, muz = shear_avg_4th_order(mu, nx, nz, mux, muz)
  elif accuracy == 6:
      mux, muz = shear_avg_6th_order(mu, nx, nz, mux, muz)
  elif accuracy == 8:
      mux, muz = shear_avg_8th_order(mu, nx, nz, mux, muz)

  solution_space = np.zeros((nx, nz, nt)) 

  my_absorber = Absorber(nx, nz, w, a)
  absorb_coeff = my_absorber.compute_absorption_coefficients()

  for it in range(nt):
    vy[isx, isz] += dt**2 * ST[it] / rho[isx, isz]

    if accuracy == 2:
      vy = update_vel_2nd_order(vy, syx, syz, dx, dz, dt, nx, nz, rho)
    elif accuracy == 4:
      vy = update_vel_4th_order(vy, syx, syz, dx, dz, dt, nx, nz, rho)
    elif accuracy == 6:
      vy = update_vel_6th_order(vy, syx, syz, dx, dz, dt, nx, nz, rho)
    elif accuracy == 8:
      vy = update_vel_8th_order(vy, syx, syz, dx, dz, dt, nx, nz, rho)
      
    if accuracy == 2:
      syx, syz = update_stress_2nd_order(vy, syx, syz, dx, dz, dt, nx, nz, mux, muz)
    elif accuracy == 4:
       syx, syz = update_stress_4th_order(vy, syx, syz, dx, dz, dt, nx, nz, mux, muz)
    elif accuracy == 6:
       syx, syz = update_stress_6th_order(vy, syx, syz, dx, dz, dt, nx, nz, mux, muz)
    elif accuracy == 8:
       syx, syz = update_stress_8th_order(vy, syx, syz, dx, dz, dt, nx, nz, mux, muz)

    vy *= absorb_coeff  
    syx *= absorb_coeff 
    syz *= absorb_coeff

    solution_space[:, :, it] = vy
  return solution_space, vy



#!/usr/bin/env python3
import numpy as np

class Absorber:
    def __init__(self, nx, nz, w, a):
        self.nx = nx
        self.nz = nz
        self.w = w
        self.a = a
        self.coeff = np.zeros(self.w)
        self.absorb_coeff = np.ones((self.nx, self.nz))
        
        for i in range(self.w):    
            self.coeff[i] = np.exp(-(self.a**2 * (self.w-i)**2))
    
    def compute_absorption_coefficients(self):
        zb = 0 
        for i in range(self.w):
            ze = self.nz - i - 1
            for j in range(zb, ze):
                self.absorb_coeff[i, j] = self.coeff[i]
      
        zb = 0
        for i in range(self.w):
            ii = self.nx - i - 1
            ze = self.nz - i - 1
            for j in range(zb, ze):
                self.absorb_coeff[ii, j] = self.coeff[i]
                
        for j in range(self.w):
            jj = self.nz - j - 1
            xb = j
            xe = self.nx - j
            for i in range(xb, xe):
                self.absorb_coeff[i, jj] = self.coeff[j]

        return self.absorb_coeff


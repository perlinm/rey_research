#!/usr/bin/env python3

from numpy import array, pi
from sympy import gamma
from matplotlib import rcParams as rc

all_colors = [ c["color"] for c in rc['axes.prop_cycle'] ]
colors = [ all_colors[ii] for ii in [0,2,3,1,6] ] # color convention by atom number

spins = [ -1, 1 ] # numerical values for pseudospins: g <--> -1 and e <--> 1


##########################################################################################
# Sr-87 optical lattice constants
##########################################################################################

### constants in SI units

hbar_SI = 6.62607015e-34 / (2*pi) # reduced planck constant
c_SI = 299792458 # speed of light
m_e_SI = 9.10938291e-31 # mass of electron
bohr_SI = 5.2917721067e-11 # bohr radius

m_SR87_SI = 1.4431558e-25 # mass of strontium-87
decay_rate_SI = 6e-3 # g --> e decay rate by spontaneous emission

l_clock_SI = 698e-9 # clock laser wavelength
l_lattice_SI = 813.4e-9 # lattice laser wavelength

k_clock_SI = 2 * pi / l_clock_SI # clock wavenumber
k_lattice_SI = 2 * pi / l_lattice_SI # lattice wavenumber
w_clock_SI = k_clock_SI * c_SI # clock angular frequency
w_lattice_SI = k_lattice_SI * c_SI # lattice angular frequency

recoil_energy_SI = (hbar_SI*k_lattice_SI)**2 / (2 * m_SR87_SI) # lattice recoil energy

### constants in natural units (NU)

m_SR87_NU = m_SR87_SI * c_SI**2 / hbar_SI

w_clock_NU = w_clock_SI
w_lattice_NU = w_lattice_SI
k_clock_NU = w_clock_NU
k_lattice_NU = w_lattice_NU

recoil_energy_NU = k_lattice_NU**2 / (2 * m_SR87_NU)
recoil_energy_Hz = recoil_energy_NU / (2*pi)

### constants in atomic units

m_SR87_AU = m_SR87_SI / m_e_SI
k_lattice_AU = k_lattice_SI * bohr_SI
recoil_energy_AU = k_lattice_AU**2 / (2 * m_SR87_AU)

### constants in "lattice units" (LU) with k_lattice = recoil_energy = hbar = 1

m_SR87_LU = 0.5 # because E_R = k_L^2 / (2 * m_SR87)
decay_rate_LU = decay_rate_SI / recoil_energy_NU

w_lattice_LU = w_lattice_NU / recoil_energy_NU
w_clock_LU = w_clock_NU / recoil_energy_NU
k_clock_LU = k_clock_NU / k_lattice_NU


##########################################################################################
# two-body collisional parameters
##########################################################################################

# s-wave scattering lengths for Sr-87 atoms in bohr (i.e. in atomic units, AU)
# a_gg and a_ee are obtained from zhang2014spectroscopic
# a_eg_p and a_eg_m are obtained from goban2018emergence
a_gg_AU = 96.2 # +/- 0.1
a_eg_m_AU = 69.1 # (0.2)stat (0.8)sys
a_eg_p_AU = 160 # (0.5)stat (4.4)sys
a_ee_AU = 176 # +/- 11
a_int_AU = array([ a_gg_AU, a_eg_m_AU, a_eg_p_AU, a_ee_AU ])

# p-wave interaction lengths for Sr-87 atoms in bohr; obtained from zhang2014spectroscopic
b_gg_AU = 74.6 # +/- 0.4
b_eg_m_AU = -42 # +103 / -22
b_eg_p_AU = -169 # +/- 23
b_ee_AU = -119 # +/- 18
b_int_AU = array([ b_gg_AU, b_eg_p_AU, b_eg_m_AU, b_ee_AU ])

a_int_LU = a_int_AU * k_lattice_AU # s-wave scattering lengths
b_int_LU = b_int_AU * k_lattice_AU # p-wave scattering lengths
g_int_LU = (4*pi/m_SR87_LU) * a_int_LU # s-wave interaction strengths
gp_int_LU = (3*pi/m_SR87_LU) * b_int_LU**3 # p-wave interaction strengths

# van der Waals interaction coefficients; obtained from zhang2014spectroscopic
C6_gg_AU = 3107
C6_eg_AU = 3880 # +/- 80
C6_ee_AU = 5360 # +/- 200
C6_AU = array([ C6_gg_AU, C6_eg_AU, C6_eg_AU, C6_ee_AU ])

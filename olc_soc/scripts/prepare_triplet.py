#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import itertools

from scipy.integrate import quad, solve_ivp

from mathieu_methods import mathieu_solution
from overlap_methods import pair_overlap_1D, tunneling_1D, lattice_wavefunction
from sr87_olc_constants import g_int_LU, recoil_energy_Hz

from fermion_methods import f_op, sum_ops, mul_ops, rank_comb, restrict_matrix

from time import time
start_time = time()

np.set_printoptions(linewidth = 200)

##########################################################################################
# free lattice parameters
##########################################################################################

V_P = 3 # "primary" lattice depth in units of "regular" recoil energy
V_T = 50 # "transverse" lattice depths in units of "regular" recoil energy
stretch_P = 1.7 # "stretch factor" for accordion lattice along primary axis
stretch_T = 1.4 # "stretch factor" for accordion lattice along transverse axis
tilt = 5e3 # 2\pi Hz per lattice site
magnetic_field = 500 # Gauss; for nuclear spin splitting
rabi_frequency = 1e3 # 2\pi Hz; clock laser

relevance_cutoff = 1e-2 # for reducing "relevant" Hilbert space

# number of lattice bands and lattice sites to use in mathieu equation solver
bands, site_number = 5, 100

##########################################################################################
# fixed and derived lattice parameters
##########################################################################################

nuclear_splitting_per_Gauss = 110 # 2\pi Hz / Gauss / nuclear spin
nuclear_splitting = magnetic_field * nuclear_splitting_per_Gauss

# lattice depths in units of "stretched" recoil energies
V_P_S = V_P * stretch_P**2
V_T_S = V_T * stretch_T**2

# solve mathieu equation along primary and transverse axes
momenta_P_S, fourier_vecs_P_S, _ = mathieu_solution(V_P_S, bands, site_number)
momenta_T_S, fourier_vecs_T_S, _ = mathieu_solution(V_T_S, bands, site_number)

# compute overlap integrals and tunneling rates in stretched recoil energies
K_P_S = pair_overlap_1D(momenta_P_S, fourier_vecs_P_S)
K_T_S = pair_overlap_1D(momenta_T_S, fourier_vecs_T_S)
tunneling_rate_P_S = tunneling_1D(V_P_S, momenta_P_S, fourier_vecs_P_S)
tunneling_rate_T_S = tunneling_1D(V_T_S, momenta_T_S, fourier_vecs_T_S)

# overlap integrals and tunneling rates in regular recoil energies
K_P = K_P_S / stretch_P
K_T = K_T_S / stretch_T
tunneling_rate_P = tunneling_rate_P_S / stretch_P**2
tunneling_rate_T = tunneling_rate_T_S / stretch_T**2

# on-site interaction energies
U_int = g_int_LU * K_T**2 * K_P

# convert relevant energies to units of 2\pi Hz
tunneling_rate_P *= recoil_energy_Hz
tunneling_rate_T *= recoil_energy_Hz
U_int *= recoil_energy_Hz

print("J_P:", tunneling_rate_P)
print("J_T:", tunneling_rate_T)
print()
print("U:", U_int[0])
print("tilt:", tilt)
print()
print("\u03A9_0:", rabi_frequency)

##########################################################################################
# compute ratio of on-site to inter-site laser coupling
##########################################################################################

# numerical integral over the entire lattice
def lattice_integral(integrand, site_number, subinterval_limit = 500):
    lattice_length = np.pi * site_number

    def real_integrand(z): return np.real(integrand(z))
    def imag_integrand(z): return np.imag(integrand(z))

    real_half = quad(real_integrand, -lattice_length/2, lattice_length/2,
                     limit = subinterval_limit)
    imag_half = quad(imag_integrand, -lattice_length/2, lattice_length/2,
                     limit = subinterval_limit)
    return real_half[0] + 1j * imag_half[0]

# compute laser-induced overlap of localized wavefunctions
def laser_overlap(site_phase, momenta, fourier_vecs, site_shift = 0):
    def integrand(z):
        return ( np.conj(lattice_wavefunction(z, 0, momenta, fourier_vecs))
                 * np.exp(1j * site_phase/np.pi * z)
                 * lattice_wavefunction(z, 0, momenta, fourier_vecs, site_shift) )
    return lattice_integral(integrand, len(momenta))

# ratio of nearest-neighbor to on-site rabi strength
rabi_ratio = ( laser_overlap(np.pi, momenta_P_S, fourier_vecs_P_S, 1) /
               laser_overlap(np.pi, momenta_P_S, fourier_vecs_P_S) )

print("\u03A9_x:", rabi_frequency * abs(rabi_ratio))

exit()

##########################################################################################
# 3-tuplet preparation parameters
##########################################################################################

sites = 3
atoms = 3

up, dn = +1, -1
spins = [ up, dn ]
spin_text = { up : "+1", dn : "-1" }

# number of single-particle states
single_states = sites * atoms * len(spins)

# map from single-particle state indices to a unique integer for that state
def index_map(indices):
    site, nuclear_spin, spin = indices
    spin = (spin + 1) // 2
    return ( site * len(spins) + spin ) * atoms + nuclear_spin

def spin_row_site(site_occupations):
    return " ".join([ "o" if site_occupied else "_"
                      for site_occupied in site_occupations ])

def spin_row_single(lattice_occupations):
    return " | ".join([ spin_row_site(site_occupations)
                        for site_occupations in lattice_occupations ])

def idx_spin(seq, spin):
    indices = [ op.index for op in seq ]
    return [ idx[:2] for idx in indices if idx[-1] == spin ]

def spin_row(spin, *seqs):
    seq_text = {}
    for seq in seqs:
        lattice_occupations = [ [ (site,atom) in idx_spin(seq, spin)
                                  for atom in range(atoms) ]
                                for site in range(sites) ]
        seq_text[seq] = spin_row_single(lattice_occupations)
    return "  ==>  ".join(seq_text.values())

def seq_text(*seqs):
    state_text = "\n".join([ spin_row(spin, *seqs) for spin in spins ])
    return "\n" + state_text + "\n"

##########################################################################################
# build Hamiltonians
##########################################################################################

### on-site and inter-site spin-flip operators

dn_to_up_onsite = sum_ops( f_op(jj,mm,up).dag() * f_op(jj,mm,dn)
                           for jj in range(sites)
                           for mm in range(atoms) )

dn_to_up_tunneling = sum_ops( f_op((jj+1)%sites,mm,up).dag() * f_op(jj,mm,dn) +
                              f_op(jj,mm,up).dag() * f_op((jj+1)%sites,mm,dn)
                              for jj in range(sites)
                              for mm in range(atoms) )

dn_to_up = dn_to_up_onsite + rabi_ratio * dn_to_up_tunneling
spin_flip = dn_to_up + dn_to_up.dag()

### interactions

def coupling(qq,rr,ss,tt):
    if qq == rr and rr == ss and ss == tt:
        return U_int[ 0 if qq is dn else -1 ]
    if qq != rr and (qq,rr) == (ss,tt):
        return ( U_int[2] + U_int[1] ) / 2
    if qq != rr and (qq,rr) == (tt,ss):
        return ( U_int[2] - U_int[1] ) / 2
    else:
        return 0

H_int = 1/2 * sum_ops( coupling(qq,rr,ss,tt) *
                       f_op(jj,mm,ss).dag() * f_op(jj,nn,tt).dag() *
                       f_op(jj,nn,rr) * f_op(jj,mm,qq)
                       for jj in range(sites)
                       for mm in range(atoms)
                       for nn in range(atoms)
                       for qq, rr, ss, tt in itertools.product(spins, repeat = 4) )

### tunneling

tunnel_right = sum_ops( f_op(jj+1,mm,ss).dag() * f_op(jj,mm,ss)
                        for jj in range(sites-1)
                        for mm in range(atoms)
                        for ss in spins )
H_tunneling = tunnel_right + tunnel_right.dag()
H_tunneling *= -tunneling_rate_P

### tilt

H_tilt = sum_ops( jj * f_op(jj,mm,ss).dag() * f_op(jj,mm,ss)
                  for jj in range(sites)
                  for mm in range(atoms)
                  for ss in spins )
H_tilt *= tilt

### magnetic field

H_mag = sum_ops( mm * ss * f_op(jj,mm,ss).dag() * f_op(jj,mm,ss)
                 for jj in range(sites)
                 for mm in range(atoms)
                 for ss in spins )
H_mag *= nuclear_splitting / 2

# net static Hamiltonian
H_0 = H_int + H_tunneling + H_tilt + H_mag

##########################################################################################
# target states in 3-tuplet preparation sequence
##########################################################################################

site_0 = sites // 2
site_L = ( site_0 - 1 ) % sites
site_R = ( site_0 + 1 ) % sites

psi_0 = mul_ops( f_op(site_0,nuclear_spin,dn).dag()
                 for nuclear_spin in range(atoms) ).sorted(atoms)
psi_1 = ( f_op(site_L,0,up).dag() * f_op(site_0,0,dn) * psi_0 ).sorted(atoms)
psi_2 = ( f_op(site_L,0,dn).dag() * f_op(site_L,0,up) * psi_1 ).sorted(atoms)
psi_3 = ( f_op(site_R,2,up).dag() * f_op(site_0,2,dn) * psi_2 ).sorted(atoms)
psi_4 = ( f_op(site_R,2,dn).dag() * f_op(site_R,2,up) * psi_3 ).sorted(atoms)

psi_X = [ psi_0, psi_1, psi_2, psi_3, psi_4 ]

##########################################################################################
# miscellaneous operators
##########################################################################################

state_pop_ops = { f"$({jj},{mm},{spin_text[ss]})$" :
                  f_op(jj,mm,ss).dag() * f_op(jj,mm,ss)
                  for jj in range(sites) for mm in range(atoms) for ss in spins }

spin_pop_ops = { spin : sum_ops( f_op(jj,mm,spin).dag() * f_op(jj,mm,spin)
                                 for jj in range(sites) for mm in range(atoms) )
                 for spin in spins }
spin_Z_op = ( spin_pop_ops[up] - spin_pop_ops[dn] ) / 2
spin_Z_vals = np.array([ ( psi.dag() * ( spin_Z_op * psi ).sorted(atoms) ).sorted(0)
                         for psi in psi_X ])
spin_changes = spin_Z_vals[1:] - spin_Z_vals[:-1]
spin_change_dirs = [ +1 if spin_change > 0 else -1 for spin_change in spin_changes ]

def energy(state):
    return ( state.dag() * ( H_0 * state ).sorted(atoms) ).sorted(0)

energies = np.array([ ( psi.dag() * ( H_0 * psi ).sorted(atoms) ).sorted(0) for psi in psi_X ])
detunings = [ sign * energy_diff
              for sign, energy_diff in zip(spin_change_dirs, energies[1:] - energies[:-1]) ]
couplings = [ ( psi_X[jj+1].dag() * ( spin_flip * psi_X[jj] ).sorted(atoms) ).sorted(0)
              for jj in range(len(detunings)) ]

##########################################################################################
# build matrix objects
##########################################################################################

allowed_states = [ rank_comb((index_map((site_0,0,spin_0)),
                              index_map((site_1,1,spin_1)),
                              index_map((site_2,2,spin_2))))
                   for site_0 in range(sites)
                   for site_1 in range(sites)
                   for site_2 in range(sites)
                   for spin_0 in spins
                   for spin_1 in spins
                   for spin_2 in spins ]
subspace_map = { index : jj for jj, index in enumerate(allowed_states) }


##########################################################################################
##########################################################################################
##########################################################################################

sim_step = 0

print()
print("start:")
print(seq_text(psi_X[sim_step].seqs[0]))
print("end:")
print(seq_text(psi_X[sim_step+1].seqs[0]))

print("-"*80,"\n")
print("relevant processes:\n")

allowed_states = { seq : 1 for seq in psi_X[sim_step].seqs }
checked_states = set()

jump_ops = [ rabi_frequency/2 * spin_flip, H_0 ]
jump_detunings = [ spin_change_dirs[sim_step] * detunings[sim_step], 0 ]

jumps = 0
new_states = True
while new_states:
    jumps += 1
    new_states = False
    for start in list(allowed_states.keys()):
        if start in checked_states: continue
        checked_states.add(start)
        start_relevance = allowed_states[start]

        for jump_op, detuning in zip(jump_ops, jump_detunings):
            laser_image = ( jump_op * start ).sorted(atoms)
            for end, coupling in laser_image:
                if end == start: continue
                assert(coupling != 0)

                static_field = ( energy(end) - energy(start) - detuning ) / 2
                if static_field == 0:
                    relevance = 1
                else:
                    weight = abs(coupling / static_field)
                    relevance = 2/np.pi * np.arctan(weight)

                net_relevance = relevance * start_relevance
                if end in allowed_states:
                    allowed_states[end] = max(net_relevance, allowed_states[end])
                    continue

                if net_relevance > relevance_cutoff:
                    print("relevance:",relevance)
                    print("net_relevance:",net_relevance)
                    print(seq_text(start,end))
                    allowed_states[end] = net_relevance
                    new_states = True

print("-"*80,"\n")
print("relevant states:",len(allowed_states))
print()
for seq, relevance in sorted(allowed_states.items(), key = lambda x : x[1]):
    print("net_relevance:",relevance)
    print(seq_text(seq))

print("jumps:",jumps)
assert(psi_X[sim_step+1].seqs[0] in allowed_states.keys())

subspace_map = { rank_comb([ index_map(op.index) for op in seq ]) : idx
                 for idx, seq in enumerate(allowed_states.keys()) }

##########################################################################################
##########################################################################################
##########################################################################################

print("subspace dimension:",len(subspace_map))
print(time()-start_time)

exit()

print()
start_time = time()


def to_matrix(operator):
    mat = operator.matrix(single_states, atoms, index_map)
    return restrict_matrix(mat, subspace_map)

def to_vector(operator):
    vec = operator.vector(single_states, index_map)
    return restrict_matrix(vec, subspace_map)

H_0 = to_matrix(H_0)
dn_to_up = to_matrix(dn_to_up)

psi_0 = to_vector(psi_0)
psi_1 = to_vector(psi_1)
psi_X = [ to_vector(psi) for psi in psi_X ]
state_pop_ops = { label : to_matrix(op) for label, op in state_pop_ops.items() }

print("built matrix objects")
print(time()-start_time)
print()
start_time = time()

##########################################################################################
# simulate time evolution
##########################################################################################

ivp_tolerance = 1e-6

pi_times = [ 1/2 / abs( rabi_frequency * coupling ) for coupling in couplings ]

def H_laser(phase):
    matrix = rabi_frequency/2 * np.exp(-1j*phase) * dn_to_up
    return matrix + matrix.conj().T

def time_deriv(step, time, state):
    return -1j * 2*np.pi * ( H_0 + H_laser(2*np.pi*detunings[step]*time) ).dot(state)

times = np.zeros(1)
states = np.array(psi_X[0].toarray().flatten().astype(complex), ndmin = 2)
for step in range(len(couplings)):

    ########################################
    if sim_step is not None:
        if step != sim_step: continue
        states = np.array(psi_X[sim_step].toarray().flatten().astype(complex), ndmin = 2)
    ########################################

    print("starting step:",step)

    solution = solve_ivp(lambda time, state : time_deriv(step, time, state),
                         (0,pi_times[step]), states[-1,:],
                         rtol = ivp_tolerance, atol = ivp_tolerance)
    times = np.append(times, times[-1] + solution.t[1:])
    states = np.vstack([ states, solution.y.T[1:,:] ])

    print(time()-start_time)
    print()
    start_time = time()

print("finished simulating")
print("experiment time:",times[-1])
print("result shape:",states.shape)
final_pops = abs(states[-1,:])**2
final_pops /= sum(final_pops)
final_pops.sort()

pop_cutoff = 1e-3
print(f"populations above {pop_cutoff}:")
print([ pop for pop in final_pops[::-1] if pop > pop_cutoff ])

##########################################################################################
# plot single-particle state populations
##########################################################################################

def expct(state, op):
    return state.conj().T @ ( op @ state ) / ( state.conj().T @ state )

state_pops = { label : np.array([ np.real( expct(state, state_pop_op) )
                                  for state in states ])
               for label, state_pop_op in state_pop_ops.items() }

plot_cutoff = 0.02
def static_pop(pop):
    return max(abs(pop-pop[0])) < plot_cutoff

for label, state_pop in state_pops.items():
    if static_pop(state_pop): continue
    plt.plot(times, state_pop, label = label)

print("computed expectation values")
print(time()-start_time)
start_time = time()

plt.xlim(times[0],times[-1])
plt.axhline(0, color = "gray", linewidth = 0.5)
plt.axhline(1, color = "gray", linewidth = 0.5)
plt.xlabel("Time (sec)")
plt.ylabel("Population")
plt.legend(loc = "best")
plt.tight_layout()
plt.show()
plt.close()

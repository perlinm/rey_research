#!/usr/bin/env python3

from pylab import *

rcParams.update({"text.usetex" : True}) # use latex math font

figure(figsize = (3,2))
plot_points = 1000

eta = linspace(1/plot_points,1,plot_points)
chi = ( 1/eta**2 * (1 + sin(2*pi*eta)/(2*pi*eta) - 2*(sin(pi*eta)/(pi*eta))**2) )
chi /= max(chi)

plot(eta, chi, "k")
xlabel(r"$\eta$")
ylabel(r"$\chi/\chi_{\mathrm{max}}$")
xlim(0,1)
tight_layout()
savefig("../figures/squeezing_strength.pdf")

print("eta_0:",eta[chi.argmax()])

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Imports:

# %%
import aerosandbox as asb
import aerosandbox.numpy as np

# from small_glide import hor_vol_coef
from wing_deflection import max_deflection

# %% [markdown]
# Constants, definitions that shouldn't change

# %%
wing_airfoil = asb.Airfoil("naca0001")
tail_airfoil = asb.Airfoil("naca0001")

E = 3e9
rho_balsa = 250 # kg/m3
rho = 1.225
nu = 1.5e-5
pi = np.pi
AR_v = 3

H_loc = 0.1 #distance of LE of horizontal stabilizer and vert stabilizer from the LE of the wing

print("blah")
# %% [markdown]
# Beginning the optimization environment, initializing some variables
#

# %%
opti = asb.Opti()

V = opti.variable(init_guess = 4, lower_bound = 0.1, upper_bound = 5)
AR = opti.variable(init_guess = 5, lower_bound = 2, upper_bound = 15)
S = opti.variable(init_guess = 0.05, lower_bound = 0.01, upper_bound = 0.5)
twist = opti.variable(init_guess = 2, lower_bound = 0, upper_bound = 10)
dihedral = opti.variable(init_guess = 5, lower_bound = 0, upper_bound=10)
AR_h = opti.variable(init_guess = 4, lower_bound = 2)
S_h = opti.variable(init_guess = 0.01, lower_bound = 0.00001)
S_v = opti.variable(init_guess = 0.005, lower_bound = 0.000001)
glide_angle = opti.variable(init_guess = 10, lower_bound=0)

# %% [markdown]
# Derived variables from the given optimized variables

# %%
b_total = np.sqrt(S * AR)
c = S / b_total
b = b_total / 2

b_h = np.sqrt(S_h * AR_h) / 2
c_h = S_h / (b_h * 2)

b_v = np.sqrt(S_v * AR_v)
c_v = S_v / b_v

# %% [markdown]
# Weight and structural model

# %%
thickness = 0.00025
boom_area = thickness * 0.01
margin = 0.001
I_formula = lambda chord, tau: chord * thickness ** 3 / 12

weight = (S + S_h + S_v) * thickness * rho_balsa + H_loc * rho_balsa * boom_area + margin

cumsum_weight = (c / 2 * S + (H_loc + c_h/2) * S_h + (H_loc + c_v/2) * S_v) * thickness * rho_balsa + H_loc / 2 * H_loc * boom_area * rho_balsa
COM = cumsum_weight / weight

# Structural Deflection Constraint (tau set to 0.05 dynamically)
N = 1.2
opti.subject_to(max_deflection(N * weight * 9.81, AR, S, E, I_formula = I_formula, tau=0.05) < 0.05)
opti.subject_to(max_deflection(N * weight * 9.81, AR_h, S_h, E, I_formula = I_formula, tau=0.05) < 0.05)
opti.subject_to(max_deflection(N * weight * 9.81, AR_v, S_v, E, I_formula = I_formula, tau=0.05) < 0.02)


# %% [markdown]
# Aerodynamic Constraints

# %%
import casadi as ca

#tail volume coefficients
hor_vol_coef = S_h * H_loc / (S * c)
ver_vol_coef = S_v * H_loc / (S * c)

opti.subject_to(hor_vol_coef > 0.3)
opti.subject_to(hor_vol_coef < 0.6)

opti.subject_to(ver_vol_coef > 0.02)
opti.subject_to(ver_vol_coef < 0.05)

#neutral point
a_w = 2 * pi / (1 + 2 / AR)
a_h = 2 * pi / (1 + 2 / AR_h)

np = c * (a_w / (4 * a_h) + hor_vol_coef * (1 + c / (4 * H_loc))) / (a_w / a_h + hor_vol_coef * c / H_loc )
opti.subject_to(np > COM)

#aerodynamic constraints

a0 = 2 * pi
e = 0.9
Re = V * c / nu

CL = a0 / (1 + a0 / (pi * AR * e)) * twist

CL_h = - (COM / c - 0.25) * CL / ((COM / H_loc - 1 - c / H_loc) * hor_vol_coef)
alpha_t = CL_h / (2 * pi)

CDi = CL**2 / (pi * AR * e) + CL_h ** 2 / (pi * AR_h * e)
CD0 = 1.3 * 2.656 / Re ** 0.5
CD = CDi + CD0

q = 1/2 * rho * V**2 * S
L = q * (CL + CL_h)
D = q * CD

opti.subject_to(weight * 9.81 * ca.cos(glide_angle * pi / 180) < L)
opti.subject_to(weight * 9.81 * ca.sin(glide_angle * pi / 180) > D)
opti.subject_to(CL < 1.4)

#spiral stability
spiral  = H_loc * dihedral / (b_total * CL)

opti.subject_to(spiral > 5)

# %% [markdown]
# Objective function and solving

# %%
opti.minimize(glide_angle)

# --- Solve the Problem ---
sol = opti.solve(behavior_on_failure="return_last", max_iter=100,
    options={"ipopt.linear_solver": "mumps",
    "ipopt.mumps_mem_percent":300,
    })


# %%

import aerosandbox as asb
import aerosandbox.numpy as np
import pickle
from wing_deflection import max_deflection

wing_airfoil = asb.Airfoil("naca0001")
tail_airfoil = asb.Airfoil("naca0001")

opti = asb.Opti()

# Material Properties
E = 3e9
rho_balsa = 200 # kg/m3

# FIX 3: Let the moment of inertia scale continuously using tau
I_formula = lambda chord, tau: chord * (chord * tau) ** 3 / 12

# --- Decision Variables ---
# FIX 1: Initial guess (4) must be lower than upper bound (5)
V = opti.variable(init_guess = 4, lower_bound = 0.1, upper_bound = 5)

AR = opti.variable(init_guess = 5, lower_bound = 2, upper_bound = 15)
S = opti.variable(init_guess = 0.05, lower_bound = 0.01, upper_bound = 0.5)
twist = opti.variable(init_guess = 2, lower_bound = 0, upper_bound = 10)

# FIX 2: Correct geometric formulas for wingspan and chord
b_total = np.sqrt(S * AR)
c = S / b_total
b = b_total / 2

dihedral = opti.variable(init_guess = 5, lower_bound = 0, upper_bound = 15)

AR_h = opti.variable(init_guess = 4, lower_bound = 2)
S_h = opti.variable(init_guess = 0.01, lower_bound = 0.001)
b_h = np.sqrt(S_h * AR_h) / 2
c_h = S_h / (b_h * 2)

AR_v = opti.variable(init_guess = 3, lower_bound = 1)
S_v = opti.variable(init_guess = 0.005, lower_bound = 0.001)
b_v = np.sqrt(S_v * AR_v) 
c_v = S_v / b_v

H_loc = opti.variable(init_guess = 0.3, lower_bound = 0.1, upper_bound = 1.0)

# --- Geometry Generation ---
airplane = asb.Airplane(
    name="Peter's Glider",
    xyz_ref=[0, 0, 0],  
    wings=[
        asb.Wing(
            name="Main Wing",
            symmetric=True,  
            xsecs=[  
                asb.WingXSec(  
                    xyz_le=[0, 0, 0],  
                    chord=c,
                    twist=twist,  
                    airfoil=wing_airfoil,  
                ),
                asb.WingXSec(  
                    xyz_le=[0, b, b * dihedral * np.pi / 180],
                    chord=c,
                    twist=twist,
                    airfoil=wing_airfoil,
                ),
            ],
        ),
        asb.Wing(
            name="Horizontal Stabilizer",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=c_h,
                    twist=0,
                    airfoil=tail_airfoil,
                ),
                asb.WingXSec(
                    xyz_le=[0, b_h, 0], chord=c_h, twist=0, airfoil=tail_airfoil
                ),
            ],
        ).translate([H_loc, 0, 0]),
        asb.Wing(
            name="Vertical Stabilizer",
            symmetric=False,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=c_v,
                    twist=0,
                    airfoil=tail_airfoil,
                ),
                asb.WingXSec(
                    xyz_le=[0, 0, b_v], chord=c_v, twist=0, airfoil=tail_airfoil
                ),
            ],
        ).translate([H_loc, 0, 0]),
    ],
)

# --- Weight and Aerodynamics ---
weight = (S + S_h + S_v) * 0.003 * rho_balsa + H_loc * rho_balsa * 0.00001 + 0.01

vlm = asb.VortexLatticeMethod(
    airplane=airplane,
    op_point=asb.OperatingPoint(
        velocity=V,  
        alpha=0,  
    ),
)
aero = vlm.run()


# --- Optimization Constraints ---
opti.subject_to(weight * 9.81 < aero["L"])
opti.subject_to(aero["CL"] < 1.4)
N = 1.2
# opti.subject_to(aero["Cma"] > 0)

hor_vol_coef = S_h * H_loc / (S * c)
ver_vol_coef = S_v * H_loc / (S * c)

opti.subject_to(hor_vol_coef > 0.3)
opti.subject_to(hor_vol_coef < 0.6)

opti.subject_to(ver_vol_coef > 0.02)
opti.subject_to(ver_vol_coef < 0.05)

spiral  = H_loc * dihedral / (b_total * aero["CL"])

opti.subject_to(spiral > 5)

# Structural Deflection Constraint (tau set to 0.05 dynamically)
opti.subject_to(max_deflection(N * weight * 9.81, AR, S, E, I_formula = I_formula, tau=0.05) < 0.05)
opti.subject_to(max_deflection(N * weight * 9.81, AR_h, S_h, E, I_formula = I_formula, tau=0.05) < 0.05)
opti.subject_to(max_deflection(N * weight * 9.81, AR_v, S_v, E, I_formula = I_formula, tau=0.05) < 0.02)

# Scaled Objective Function
opti.minimize(weight * 1000)

# --- Solve the Problem ---
sol = opti.solve(behavior_on_failure="return_last", max_iter=100, 
    options={"ipopt.linear_solver": "mumps",
    "ipopt.mumps_mem_percent":300,
    
    })

# --- Cleaned Print Statements ---
print("\n" + "="*40)
print("       Grass'S GLIDER DESIGN METRICS     ")
print("="*40)
print(f"Optimization Status : {sol.stats()['return_status']}")
print(f"Total Glider Mass   : {sol.value(weight) * 1000:.2f} grams")
print(f"Design Velocity     : {sol.value(V):.2f} m/s")
print("-"*40)
print(f"Main Wing Area (S)  : {sol.value(S):.4f} m^2")
print(f"Aspect Ratio (AR)   : {sol.value(AR):.2f}")
print(f"Total Wingspan (b)  : {sol.value(b_total):.2f} m")
print(f"Wing Chord (c)      : {sol.value(c)*1000:.1f} mm")
print(f"Wing Dihedral       : {sol.value(dihedral):.1f} degrees")
print(f"Wing Twist          : {sol.value(twist):.1f} degrees")
print("-"*40)
print(f"Horizontal Tail Area: {sol.value(S_h):.4f} m^2")
print(f"Vertical Tail Area  : {sol.value(S_v):.4f} m^2")

print(f"Horizontal Tail AR: {sol.value(AR_h):.4f} m^2")

print(f"Vertical Tail AR: {sol.value(AR_v):.4f} m^2")
print(f"Tail Boom Length    : {sol.value(H_loc):.2f} m")
print("-"*40)
print(f"Aerodynamic Lift    : {sol.value(aero['L']):.3f} N")
print(f"Required Weight Lift: {sol.value(weight)*9.81:.3f} N")
print(f"Operating CL        : {sol.value(aero['CL']):.3f}")
print("-"*40)
print(f"Horiz. Vol. Coeff.  : {sol.value(hor_vol_coef):.3f}  (Target: 0.3 - 0.6)")
print(f"Vert. Vol. Coeff.   : {sol.value(ver_vol_coef):.3f}  (Target: 0.02 - 0.05)")
print(f"Spiral Criterion (B): {sol.value(spiral):.3f}")

# --- 3D Interactive Plotting ---
# Reconstruct the optimized airplane using the evaluated solution values
optimized_airplane = sol.value(airplane)

# This will open up an interactive browser window showing your full 3D layout!
optimized_airplane.draw()

# 1. Extract the raw numeric values into a clean dictionary
design_results = {
    "mass_grams": float(sol.value(weight) * 1000),
    "velocity": float(sol.value(V)),
    "S": float(sol.value(S)),
    "AR": float(sol.value(AR)),
    "wingspan": float(sol.value(b_total)),
    "chord": float(sol.value(c)),
    "dihedral": float(sol.value(dihedral)),
    "twist": float(sol.value(twist)),
    "S_h": float(sol.value(S_h)),
    "S_v": float(sol.value(S_v)),
    "H_loc": float(sol.value(H_loc)),
    "lift": float(sol.value(aero["L"])),
    "CL": float(sol.value(aero["CL"])),
}

# 2. Save the dictionary to a file using pickle
with open("glider_solution.pkl", "wb") as f:
    pickle.dump(design_results, f)

print("Solution successfully saved to 'glider_solution.pkl'!")
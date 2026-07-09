import aerosandbox.numpy as np

def asb_casadi_cumsum(vector):
    """
    A pure sequential accumulator that completely avoids NumPy's internals.
    Uses .shape[0] instead of len() to remain 100% compatible with CasADi MX types.
    """
    # Grab the dimension dynamically from the CasADi symbol shape
    n = vector.shape[0]
    
    cumulative = [vector[0]]
    for i in range(1, n):
        cumulative.append(cumulative[-1] + vector[i])
    return np.array(cumulative)

def asb_forward_cumulative_trapezoid(y, x, initial=0):
    """Integrates forward from index 0 to N-1 (Root to Tip) safely for CasADi"""
    dx = np.diff(x)
    step_areas = dx * (y[:-1] + y[1:]) / 2
    
    # Use our custom symbolic loop tracker
    cumulative = asb_casadi_cumsum(step_areas)
    
    return np.concatenate([np.array([initial]), cumulative])

def asb_backward_cumulative_trapezoid(y, x):
    """Integrates backward from N-1 to 0 (Tip to Root) safely for CasADi"""
    dx = np.diff(x)
    step_areas = dx * (y[:-1] + y[1:]) / 2
    
    total_sum = np.sum(step_areas)
    # Use our custom symbolic loop tracker
    forward_cumsum = asb_casadi_cumsum(step_areas)
    
    return np.concatenate([np.array([total_sum]), total_sum - forward_cumsum])

def max_deflection(L, AR, S, E, I_formula=None, rho = 0,  lam=1.0, tau=0.1, num_pts=20):
    b = np.sqrt(S * AR)  # Total wingspan (m)
    c_mean = S / b       # Mean chord (m)
    
    c_root = (2 * c_mean) / (1 + lam)
    c_tip = c_root * lam

    if I_formula is None:
        I_root = 0.033 * c_root * (c_root * tau) ** 3
        I_tip  = 0.033 * c_tip  * (c_tip * tau) ** 3
    else:
        I_root = I_formula(c_root, tau)
        I_tip  = I_formula(c_tip, tau)

    # Note: Keep the discretization grid strictly numeric (don't optimize num_pts!)
    x = np.linspace(0, b / 2, num_pts)

    ellipse_b = L * 4 / (np.pi * b)
    eps = 1e-12
    q_ellipse = ellipse_b * np.sqrt(np.maximum(1 - (x / (b / 2)) ** 2, eps)) 
    
    loading_per = L / S
    q_chord = loading_per * c_root * (1 - (1 - lam) * (x / (b / 2))) 

    q_weight = - (c_root * (1 - (1 - lam) * (x / (b / 2)))) ** 2  * rho * 9.81 * tau 

    q_schrenk = (q_ellipse + q_chord) / 2 

    #including the weight of the wing
    q_schrenk = q_schrenk + q_weight
    EI = E * (I_root - (I_root - I_tip) * (x / (b / 2)))

    # --- Structural Integrations ---
    # Shear Force (Tip to Root)
    shear = asb_backward_cumulative_trapezoid(q_schrenk, x)

    # Bending Moment (Tip to Root)
    moment = asb_backward_cumulative_trapezoid(shear, x)
    
    # Slope (Root to Tip)
    M_over_EI = moment / EI
    slope = asb_forward_cumulative_trapezoid(M_over_EI, x)
    
    # Deflection (Root to Tip)
    deflection = asb_forward_cumulative_trapezoid(slope, x)

    return deflection[-1]

if __name__ == "__main__":
    L = 1
    AR = 2
    S = 0.06 * 0.12
    E = 3e9
    I_formula = lambda c, tau: c * 0.00025 ** 3 / 12
    rho = 250
    print(max_deflection(L, AR, S, E, I_formula = I_formula, rho = rho, tau = 0.00025 / 0.06))
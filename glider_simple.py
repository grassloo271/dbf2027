import numpy as npt
from scipy.optimize import minimize
from wing_deflection import max_deflection
#a 10 cm long plane made of paper, hopefully everythign is a flat plate

#variables: AR, S, AR_h, S_h, AR_v, S_v, alpha

def find_CL(variables):
	e = 0.8
	a0 = 2 * np.pi
	return a0 / ( 1 + a0 / (np.pi * variables[0] * e)) * variables[6]

def objective(variables):
	AR, S, AR_h, S_h, AR_v, S_v = variables

	return alpha_min

def find_velocity(variables):
	rho = 1.225
	q = 1 / 2 * variables[1] * find_CL(variables) * rho
	return np.sqrt(find_weight(variables) / q)

def find_deflection(variables):


def find_weight(variables):
	density = 250
	return (S + S_h + S_v) * 0.00026 * rho + 0.01 * 0.00025 * rho





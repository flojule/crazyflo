#!/usr/bin/env python

import numpy as np
import argparse
import scipy.optimize

import uav_trajectory

# computes the difference between current interpolation and desired values
def func(coefficients, times, values, piece_length):
  result = 0
  i = 0
  for t, value in zip(times, values):
    if t > (i+1) * piece_length:
      i = i + 1
    estimate = np.polyval(coefficients[i*8:(i+1)*8], t - i * piece_length)
    result += (value - estimate) ** 2 #np.sum((values - estimates) ** 2)
  return result


def func_eq_constraint_der(coefficients, i, piece_length, order):
  result = 0
  last_der = np.polyder(coefficients[(i-1)*8:i*8], order)
  this_der = np.polyder(coefficients[i*8:(i+1)*8], order)

  end_val = np.polyval(last_der, piece_length)
  start_val = np.polyval(this_der, 0)
  return end_val - start_val

def func_eq_constraint_der_value(coefficients, i, t, desired_value, order):
  result = 0
  der = np.polyder(coefficients[i*8:(i+1)*8], order)

  value = np.polyval(der, t)
  return value - desired_value

def generate_trajectory(data, num_pieces):
  piece_length = data[-1,0] / num_pieces

  x0 = np.zeros(num_pieces * 8)

  constraints = []
  # piecewise values and derivatives have to match
  for i in range(1, num_pieces):
    for order in range(0, 4):
      constraints.append({'type': 'eq', 'fun': func_eq_constraint_der, 'args': (i, piece_length, order)})

  # zero derivative at the beginning and end
  for order in range(1, 3):
    constraints.append({'type': 'eq', 'fun': func_eq_constraint_der_value, 'args': (0, 0, 0, order)})
    constraints.append({'type': 'eq', 'fun': func_eq_constraint_der_value, 'args': (num_pieces-1, piece_length, 0, order)})


  resX = scipy.optimize.minimize(func, x0, (data[:,0], data[:,1], piece_length), method="SLSQP", options={"maxiter": 1000}, 
    constraints=constraints
    )
  resY = scipy.optimize.minimize(func, x0, (data[:,0], data[:,2], piece_length), method="SLSQP", options={"maxiter": 1000}, 
    constraints=constraints
    )
  resZ = scipy.optimize.minimize(func, x0, (data[:,0], data[:,3], piece_length), method="SLSQP", options={"maxiter": 1000}, 
    constraints=constraints
    )

  resYaw = scipy.optimize.minimize(func, x0, (data[:,0], data[:,4], piece_length), method="SLSQP", options={"maxiter": 1000}, 
    constraints=constraints
    )

  traj = uav_trajectory.Trajectory()
  traj.polynomials = [uav_trajectory.Polynomial4D(
    piece_length, 
    np.array(resX.x[i*8:(i+1)*8][::-1]),
    np.array(resY.x[i*8:(i+1)*8][::-1]),
    np.array(resZ.x[i*8:(i+1)*8][::-1]),
    np.array(resYaw.x[i*8:(i+1)*8][::-1])) for i in range(0, num_pieces)]
    
  traj.duration = data[-1,0]
  return traj


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input", type=str, help="CSV file containing time waypoints")
  parser.add_argument("output", type=str, help="CSV file containing trajectory with updated yaw")
  parser.add_argument("--pieces", type=int, default=5, help="number of pieces")
  args = parser.parse_args()

  data = np.loadtxt(args.input, delimiter=',', skiprows=1)
  traj = generate_trajectory(data, args.pieces)
  traj.savecsv(args.output)

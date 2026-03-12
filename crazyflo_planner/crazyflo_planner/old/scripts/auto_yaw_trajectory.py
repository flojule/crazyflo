#!/usr/bin/env python

import numpy as np
import argparse
import scipy.optimize

import uav_trajectory

def func(coefficients, tss, yawss):
  result = 0
  for ts, yaws, i in zip(tss, yawss, range(0, len(tss))):
    yaws_output = np.polyval(coefficients[i*8:(i+1)*8], ts)
    result += np.sum((yaws - yaws_output) ** 2)
  return result

def func_eq_constraint_val(coefficients, i, tss, yawss):
  result = 0
  end_val = np.polyval(coefficients[(i-1)*8:i*8], tss[i-1][-1])
  start_val = np.polyval(coefficients[i*8:(i+1)*8], tss[i][0])
  return end_val - start_val

def func_eq_constraint_der(coefficients, i, tss, yawss):
  result = 0
  last_der = np.polyder(coefficients[(i-1)*8:i*8])
  this_der = np.polyder(coefficients[i*8:(i+1)*8])

  end_val = np.polyval(last_der, tss[i-1][-1])
  start_val = np.polyval(this_der, tss[i][0])
  return end_val - start_val

def func_eq_constraint_der_value(coefficients, i, t, desired_value):
  result = 0
  der = np.polyder(coefficients[i*8:(i+1)*8])

  value = np.polyval(der, t)
  return value - desired_value

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("trajectory", type=str, help="CSV file containing trajectory")
  parser.add_argument("output", type=str, help="CSV file containing trajectory with updated yaw")
  parser.add_argument("--num", type=int, default=20, help="number of sampled points per trajectory segment")
  args = parser.parse_args()

  traj = uav_trajectory.Trajectory()
  traj.loadcsv(args.trajectory)

  tss = []
  yawss = []
  for p in traj.polynomials:
    ts = np.linspace(0, p.duration, args.num) #np.arange(0, p.duration, args.dt)
    evals = np.empty((len(ts), 15))
    for t, i in zip(ts, range(0, len(ts))):
      e = p.eval(t)
      evals[i, 0:3]  = e.pos
      evals[i, 3:6]  = e.vel
      evals[i, 6:9]  = e.acc
      evals[i, 9:12] = e.omega
      evals[i, 12]   = e.yaw
      evals[i, 13]   = e.roll
      evals[i, 14]   = e.pitch

    yaws = np.arctan2(evals[:,4], evals[:,3])
    tss.append(ts)
    yawss.append(yaws)

  x0 = np.zeros(len(traj.polynomials) * 8)
  print(x0)

  constraints = []
  for i in range(1, len(tss)):
    constraints.append({'type': 'eq', 'fun': func_eq_constraint_val, 'args': (i, tss, yawss)})
    constraints.append({'type': 'eq', 'fun': func_eq_constraint_der, 'args': (i, tss, yawss)})

  # zero derivative at the beginning and end
  constraints.append({'type': 'eq', 'fun': func_eq_constraint_der_value, 'args': (0, tss[0][0], 0)})
  constraints.append({'type': 'eq', 'fun': func_eq_constraint_der_value, 'args': (len(tss)-1, tss[-1][-1], 0)})


  res = scipy.optimize.minimize(func, x0, (tss, yawss), method="SLSQP", options={"maxiter": 1000}, 
    constraints=constraints
    )
  print(res)

  for i,p in enumerate(traj.polynomials):
    result = res.x[i*8:(i+1)*8]
    p.pyaw.p = np.array(result[::-1])

  traj.savecsv(args.output)

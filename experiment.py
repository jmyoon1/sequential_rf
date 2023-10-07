import numpy as np

eps = 0.00-1
latency = 0.1
t = np.linspace(0.001, 1, 10)

# t_low: [eps, 0.5] --> [eps, latency], [0.5, 1] --> [latency, 1], low-resolution time of VP-type SDE
# t_high: [eps, 0.5] --> [eps, 1], [0.5, 1] --> [1, 1], high-resolution time of VP-type SDE
t_low = np.maximum((latency - eps) / (1. / 2 - eps) * (t - 1. / 2) + latency, (1. - latency) / (1. / 2) * (t - 1.) + 1.)
t_high = np.minimum((1. - eps) / (1. / 2 - eps) * (t - 1. / 2) + 1., 1. * np.ones_like(t))
# t_vp = (t - eps) * (sde_rf.T - eps) / (1. / 2 - eps) + eps 
t_mask_rf = np.heaviside(1. / 2 - t, 0.5) # 1 at t = [0, sde.T / 2], 0 at t = [sde.T / 2, sde.T]
t_mask_vp = np.heaviside(t - 1. / 2, 0.5) # 0 at t = [0, sde.T / 2], 1 at t = [sde.T / 2, sde.T]

print(t_low) # [0.1, 1] at t = [0.5, 1]
print(t_high) # [1, 1] at t = [0.5, 1]
print(t_mask_rf) # 1 if t<0.5 otherwise 0
print(t_mask_vp) # 0 if t<0.5 otherwise 1
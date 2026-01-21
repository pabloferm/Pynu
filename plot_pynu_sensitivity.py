#!/usr/bin/env python
"""Plot Pynu sensitivity results with correct grid ordering."""
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Read results
hdf5_file = 'results/ORCA_sensitivity_0116.hdf5'

with h5py.File(hdf5_file, 'r') as f:
    chi2 = f['Analysis/Chi2 Systs.'][:]
    sin2theta23 = f['Physics Parameters/3-Osc/Sin2Theta23'][:]
    dm231 = f['Physics Parameters/3-Osc/Dm231'][:]

# Get unique values and grid dimensions
unique_theta = np.unique(sin2theta23)
unique_dm = np.unique(dm231)
n_theta = len(unique_theta)
n_dm = len(unique_dm)

print(f"Grid: {n_theta} theta x {n_dm} dm = {n_theta * n_dm} points")

# Data is ordered: theta varies slowest, dm varies fastest
# So reshape is (n_theta, n_dm)
chi2_2d = chi2.reshape(n_theta, n_dm)

# Handle negative chi2 (numerical issues) - set to small positive
chi2_2d[chi2_2d < 0] = 0.001

# Find minimum and compute delta chi2
min_chi2 = chi2_2d.min()
min_idx = np.unravel_index(np.argmin(chi2_2d), chi2_2d.shape)
best_theta = unique_theta[min_idx[0]]
best_dm = unique_dm[min_idx[1]]
print(f"Minimum chi2: {min_chi2:.6f} at theta={best_theta:.4f}, dm={best_dm:.6e}")

delta_chi2 = chi2_2d - min_chi2

# True values
true_theta = 0.572
true_dm = 2.511e-3
print(f"True values: theta={true_theta}, dm={true_dm:.6e}")

# Find closest grid point to true values
theta_idx = np.argmin(np.abs(unique_theta - true_theta))
dm_idx = np.argmin(np.abs(unique_dm - true_dm))
print(f"Closest to true: theta={unique_theta[theta_idx]:.4f}, dm={unique_dm[dm_idx]:.6e}, chi2={chi2_2d[theta_idx, dm_idx]:.6f}")

# Create meshgrid for plotting (theta on x-axis, dm on y-axis)
THETA, DM = np.meshgrid(unique_theta, unique_dm)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: 2D contour plot
ax1 = axes[0]
# Note: transpose because meshgrid gives (dm, theta) but data is (theta, dm)
levels = np.linspace(0, delta_chi2.max(), 20)
contour = ax1.contourf(THETA, DM * 1e3, delta_chi2.T, levels=levels, cmap='viridis')
plt.colorbar(contour, ax=ax1, label=r'$\Delta\chi^2$')

# Add confidence level contours for 2 DOF
cl_levels = [2.30, 4.61, 6.18, 9.21]  # 68%, 90%, 95%, 99%
cl_labels = {2.30: '68%', 4.61: '90%', 6.18: '95%', 9.21: '99%'}
valid_levels = [l for l in cl_levels if l < delta_chi2.max()]
if valid_levels:
    cs = ax1.contour(THETA, DM * 1e3, delta_chi2.T, levels=valid_levels, 
                     colors=['white', 'yellow', 'orange', 'red'][:len(valid_levels)], linewidths=2)
    ax1.clabel(cs, inline=True, fontsize=10, fmt=cl_labels)

# Mark best fit point
ax1.scatter([best_theta], [best_dm * 1e3], marker='*', s=200, c='cyan', edgecolors='black', 
            zorder=10, label=f'Best fit')

# Mark true values
ax1.scatter([true_theta], [true_dm * 1e3], marker='x', s=150, c='red', linewidths=3,
            zorder=10, label=f'True')

ax1.set_xlabel(r'$\sin^2\theta_{23}$', fontsize=12)
ax1.set_ylabel(r'$\Delta m^2_{31}$ [$10^{-3}$ eV$^2$]', fontsize=12)
ax1.set_title('Pynu ORCA Sensitivity (1 yr, with systematics)', fontsize=14)
ax1.legend(loc='upper left')

# Right: 1D projections
ax2 = axes[1]

# Profile over dm231 (minimize for each theta23)
chi2_theta_profile = np.min(delta_chi2, axis=1)
ax2.plot(unique_theta, chi2_theta_profile, 'b-', linewidth=2, label=r'$\Delta\chi^2(\sin^2\theta_{23})$')

# Profile over theta23 (minimize for each dm231) - on twin axis
ax2_twin = ax2.twinx()
chi2_dm_profile = np.min(delta_chi2, axis=0)
ax2_twin.plot(unique_dm * 1e3, chi2_dm_profile, 'r--', linewidth=2, label=r'$\Delta\chi^2(\Delta m^2_{31})$')
ax2_twin.set_ylabel(r'$\Delta\chi^2$ (dm profile)', fontsize=12, color='red')

# Add horizontal lines for confidence levels (1 DOF)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label=r'$\Delta\chi^2=1$ (1$\sigma$)')
ax2.axhline(y=2.71, color='gray', linestyle=':', alpha=0.7, label=r'$\Delta\chi^2=2.71$ (90%)')

ax2.axvline(x=true_theta, color='red', linestyle='-', alpha=0.5, label='True value')

ax2.set_xlabel(r'$\sin^2\theta_{23}$', fontsize=12)
ax2.set_ylabel(r'$\Delta\chi^2$ (theta profile)', fontsize=12, color='blue')
ax2.set_title('Profiled 1D Projections', fontsize=14)
ax2.legend(loc='upper left')
ax2.set_ylim(0, max(3, chi2_theta_profile.max() * 1.2))

plt.tight_layout()
plt.savefig('results/pynu_sensitivity_fixed.png', dpi=150, bbox_inches='tight')
plt.savefig('results/pynu_sensitivity_fixed.pdf', bbox_inches='tight')
print("\nSaved: results/pynu_sensitivity_fixed.png")

# Print chi2 at a few key points for comparison
print("\n=== Chi2 at key points (for cross-comparison) ===")
sample_points = [
    (0.40, 2.3e-3), (0.50, 2.5e-3), (0.572, 2.511e-3), (0.60, 2.7e-3),
    (0.45, 2.4e-3), (0.55, 2.6e-3)
]
for theta_val, dm_val in sample_points:
    t_idx = np.argmin(np.abs(unique_theta - theta_val))
    d_idx = np.argmin(np.abs(unique_dm - dm_val))
    print(f"  theta={unique_theta[t_idx]:.4f}, dm={unique_dm[d_idx]:.6e}: chi2={chi2_2d[t_idx, d_idx]:.6f}")

import matplotlib.pyplot as plt
import numpy as np

# Data from fine-tuning scan (fluct=0.0174)
lambdas = np.array([0.25, 0.28, 0.30, 0.31, 0.32, 0.35, 0.38, 0.40])
ratios = np.array([1227.0, 2094.0, 1699.0, 1696.0, 1552.0, 2090.0, 2035.0, 2258.0])
errors = np.array([215.0, 553.0, 527.0, 251.0, 124.0, 596.0, 228.0, 472.0])

# Plot setup
plt.figure(figsize=(10, 6))
plt.errorbar(lambdas, ratios, yerr=errors, fmt='o-', capsize=5, label='Simulated Ratio', color='blue', linewidth=2)

# Theoretical Target
plt.axhline(y=1836.15, color='red', linestyle='--', linewidth=2, label='Standard Model (1836.15)')

# Highlight Transition Region
plt.axvspan(0.30, 0.32, color='green', alpha=0.1, label='Stable Transition Region')

# Labels and Styling
plt.title('Geometric Mass Ratio ($m_B / m_L$) vs Core Size ($\lambda$)\nPhysical Vacuum Scale (Fluct = 1.74%)', fontsize=14)
plt.xlabel('Skyrmion Core Size ($\lambda_{decay}$)', fontsize=12)
plt.ylabel('Mass Ratio', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Annotations
plt.annotate('Stable Dip\n(1550-1700x)', xy=(0.31, 1550), xytext=(0.31, 1300),
             arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

plt.annotate('Target Crossing\n(~0.27, ~0.29)', xy=(0.28, 1836), xytext=(0.26, 2200),
             arrowprops=dict(facecolor='red', shrink=0.05), ha='center')

# Save
output_path = 'mass_ratio_optimization_plot.png'
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")

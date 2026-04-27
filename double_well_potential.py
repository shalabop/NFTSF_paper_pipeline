import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.linspace(-1.5, 1.5, 1000)
U = 2 * (x**2 - 1)**2  # double-well potential

# Figure
plt.figure(figsize=(7.5, 4.8))

# Plot curve
plt.plot(x, U, color='blue', linewidth=2)

# Limits
plt.xlim(-1.5, 1.5)
plt.ylim(0, 2)

# Labels
plt.xlabel(r'Position $(x)$', fontsize=16)
plt.ylabel(r'Potential $U(x)$', fontsize=16)

# Ticks
plt.xticks(np.arange(-1.5, 1.6, 0.5))
plt.yticks([0, 1, 2])

# Dashed vertical line at x = 0
plt.plot([0, 0], [0, 2], 'k--', linewidth=2)

# Dashed "box"
plt.plot([-1, -1], [0, 1], 'k--', linewidth=1.5)
plt.plot([-1, 0], [1, 1], 'k--', linewidth=1.5)

# Annotations
plt.text(-0.6, 1.05, r'$a$', fontsize=18)
plt.text(0.08, 1.0, r'$h$', fontsize=18)

# Style tweaks
plt.grid(False)
plt.tight_layout()

# Save image
plt.savefig("double_well_exact.png", dpi=300)

# Show
plt.show()
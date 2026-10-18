# Randomized low-rank approximation with power iterations
Implements the Halko–Martinsson–Tropp randomized range finder / SVD and validates it numerically.
**Checks:** exact recovery of low-rank matrices; empirical E||A−QQᵀA||_F against the theoretical bound;
monotone spectral-error reduction with power iterations q=0,1,2; singular-value accuracy.
Run: `pip install numpy pytest && pytest`

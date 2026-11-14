# Fixed two-source paid error-response diagnostic

Use only original K sources 0 and 4. Keep original one-shot P8 coarse solve, TSA64, outer tolerance 1e-5 and inner tolerance 1e-4. Capture actual residual after the accepted coarse injection and after second classical call. The diagnostic solver changes only its observer payload to include the actual iterate; verify r=q-Ax.

For each of the four residuals independently solve the original complex128 equation A e=r to relative residual <=1e-10 using ordinary TSA64-preconditioned FGMRES; require coercivity-based relative response-error bound <=1e-6. This is paid offline reference work, not a free deployable correction or new speed claim.

Compare orthonormal tangent Fourier energy bands 0-8,9-16,17-32,33-64 in r and e. Report every stored angular direction and Q8 energy by angle; report small tangential velocity and small normal velocity groups separately at abs(velocity)<=1e-6. Compute cross-block action P A Q e and Q A P e and cancellation, not a same-frequency assumption. Fixed cutoffs, no new source or threshold search.

Charge all input checks, operator/TSA setup, coarse factors, original-A0 qualification, incumbent solves and checks, four response solves and checks, FFT/block diagnostics, tensor saves/hashes, imports and other overhead in an exclusive total process ledger. Retain negative results and all four states. No training, heldout use, Library upload, or production modifications. One CPU thread pinned to historical comparison CPU0, GPU disabled, no locks or interruption of other research.

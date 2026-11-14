# Full-image whitening for the unresolved q3 oracle

Implementation and math note, 2026-10-02. This is a bounded numerical refinement of the same full-spatial q3 least-squares range diagnostic, not a new angular range, trained model, physical operator, or solver baseline. The eight preceding CGLS contexts reached the 100-step cap with reported relative normal residuals 0.032–0.388 and vacuous raw lower bounds. Those outcomes were underresolved; they did not exclude the angular range. No new GPU work, project-source sampling, fine teacher, training, or seed 9626 access was performed here.

`image_whitening.py` supplies `ImageWhitening(coarse)`, with `solve`, `adjoint_solve`, `matrix` and `minimum_reference_eigenvalue` for the root's frozen follow-up. `test_image_whitening.py` contains seven focused CPU tests. They passed on existing local Python/Torch 2.8.0 in 0.285 s of suite time; command:

    PYTHONPYCACHEPREFIX=/tmp/rte_whitening_pycache python3 deliverables/headroom/test_image_whitening.py

Tests verify the whitening identity, physical image and adjoints against native FP64/promoted stored FP32 geometry, original Nyquist handling, validation, broadcasting, complex linearity, and cache refresh. N8/N16 fixtures and additional angular-count fixtures are unit checks, not q5 project experiments or held-out data. GPU setup cost, large-grid accuracy and convergence remain for the root's bounded run.

## Actual homogeneous image and SPD normal matrix

Use the supplied physical operator's assembled FP64 streaming symbol, formed from its promoted stored factors, and its actual P=phase_matrix diag(weights). Let mean_t and mean_s be that material's spatial means, and let Q retain its actual Gram G=Q*Q. For every original spatial frequency,

    H0(k) = diag(actual_stream(k)) + mean_t I - mean_s P
    B0(k) = H0(k) Q                     (L-by-q)
    N0(k) = B0(k)* B0(k)                (q-by-q)
    W(k)  = N0(k)^(-1/2).

The code forms B0 **before** angular restriction, including streaming images outside range(Q). It neither regenerates ideal physical geometry nor assumes diagonal/symmetric phase. With mu=lambda_max(Herm P) and m0=mean_t-mu mean_s>0, Re<x,H0 x> >=m0||x||². Thus

    ||B0 c|| >= m0 ||Q c|| >= m0 sqrt(lambda_min(G)) ||c||,

so N0 is SPD and lambda_min(N0)>=m0² lambda_min(G). This is an exact-arithmetic statement under the stated positive-margin and skew-streaming assumptions. Every computed eigenvalue is checked finite and positive before inversion; **no eigenvalue clipping or regularization** is applied. Failure stops construction. Only the mathematically Hermitian normal Gram receives `(N0+N0*)/2` roundoff cleanup; the physical H and phase are untouched.

For exact W, `(B0 W)*(B0 W)=I`. Fourier Parseval then makes the homogeneous full-image operator H0 Q W an isometry and its normal operator identity. The saved small CPU tests evaluate this property in ordinary FP64; they are not a directed-rounding certificate.

The aliases `matrix`/`whitening_symbol` store W in material_batch-by-H-by-W-by-q-by-q format; `full_reference_symbol` stores B0 in material_batch-by-H-by-W-by-L-by-q format. The q-channel action is a spatial FFT, matrix multiply, and inverse FFT. Its adjoint uses the actual cached W*, even though the ideal inverse square root is Hermitian. Cache changes propagate through the existing coarse operator's stored-buffer signature. The implementation retains B0 for verification: at q3/L12/N256, B0, W and real eigenvalues alone occupy approximately 36+9+1.5=46.5 MiB, excluding the existing coarse cache and temporaries. Charge this setup/storage rather than treating the preconditioner as free.

## Why the old angular inverse can precondition least squares poorly

Let Aq0=Q*H0 Q, and let Pi_Q=Q G^-1 Q* be the angular orthogonal projector. The exact decomposition is

    B0*B0 = Aq0* G^-1 Aq0 + B0*(I-Pi_Q)B0.

The second term is positive semidefinite. It measures full physical images outside the retained angular range; the homogeneous angular Galerkin equation does not contain it. Therefore Aq0^-1 is a valid reduced-equation preconditioner, but it need not scale the physical least-squares image well. Full-image whitening uses both terms.

An elementary **ideal** equispaced-circle example explains the high-frequency issue. Take real q3 basis 1, sqrt(2)cos(theta), sqrt(2)sin(theta), normalized over angles, and a frequency aligned with x of magnitude kappa. The compressed streaming matrix has a zero streaming direction in the sine channel, although full streaming maps that channel to the omitted sine-second-harmonic with amplitude kappa/2. Its image norm therefore grows as kappa/2. The leading full-image normal matrix in this basis is

    kappa² diag(1/2, 3/4, 1/4).

An angular Galerkin inverse scales the sine channel by its collision rate, leaving this full-image growth unresolved; the new W scales all three channels like 1/|kappa| in this ideal large-frequency limit. This example diagnoses a plausible conditioning mechanism. The implementation uses the actual rounded geometry and finite frequencies, and this asymptotic example is not a quantitative claim about its condition number or the cause of every previous cap.

## Limits and unchanged follow-up interpretation

For heterogeneous H=H0+E, write C=HQW=C0+EQW. If delta=||EQW||2<1, singular values of C lie between 1-delta and1+delta, giving a conditional normal-equation condition bound `[(1+delta)/(1-delta)]²`. Delta has not been established here; when it is large this bound supplies no useful prediction. The original physical margin may ensure invertibility without uniform conditioning. Projection against old real AZ can additionally remove rank. **There is no universal heterogeneous, projected, 100-step or wall-time guarantee.**

The root's right-preconditioned range actions should be

    C(y)  = (I-P_AZ) H Q W(y)
    CT(v) = W* Q* H* (I-P_AZ)(v),

using the original real-stacked projector, with P_AZ absent for the raw source arm. No change of residual norm or range occurs: W is invertible, so range(QW)=range(Q). The old raw gap formula still uses the **unpreconditioned** gradient g=Q*H*(r-HQ c), original physical m, and Gram. A small preconditioned normal residual is not a replacement for that independently computed bracket. `minimum_reference_eigenvalue` describes homogeneous N0; it does not certify a lower singular value for heterogeneous HQ or AZ-projected images.

Freeze one follow-up at the same 100-step cap on the same eight contexts. Record actual W setup time/eigenmin/identity defect, adjoint check, attained physical residual, independently recomputed gradient/bracket, iteration cap/normal target and complete cached continuation timing. Compare within contemporaneous graph timing blocks. If the new solve remains capped or its raw bracket is loose, retain underresolution rather than declaring range failure. A positive finite attained proposal can witness headroom without being optimal; its actual acquisition/setup/application cost still has to fit the measured budget. No q5 solve, training, holdout access or baseline promotion follows automatically.

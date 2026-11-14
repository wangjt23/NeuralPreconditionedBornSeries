# Learn information that is expensive to discover classically

Status: architecture proposal and a bounded development oracle diagnostic,
not a trained solver or a demonstrated AI speedup. Extends the
[structure-driven plan](RTE_STRUCTURE_DRIVEN_PLAN_2026_09_13.md).
The completed diagnostic below does not support promoting the flat global
basis to a training campaign. Hierarchical block compressibility is the next
structural question; a larger network is not yet justified by these results.

## What the network should know

The strongest candidate is a **material-dependent, nonlocal response space**:
the spatial/angular patterns through which a heterogeneous medium retains and
transports errors after cheap classical corrections. Classical methods can
compute such information. The opportunity is to predict it from the material
in one amortized inference, instead of discovering it afresh through many
transport applications, calibration solves, or adaptive coarse-space setup.
This is a hypothesis to test, not an information-theoretic advantage of AI.

The distinction matters in this repository:

| Information | Cheap classical access? | Learning priority |
|---|---|---|
| Local collision inverse, streaming symbol, HSS shift | Yes, analytic here | Retain as physics |
| Local coefficient variation or a global mean | Yes | Inputs, not the main learned output |
| Which distant regions exchange slowly decaying error, with which angular structure | Requires a nonlocal response calculation in general | Main candidate |
| Exact macro Schur complement or material-adapted hard-response space | Can require nested solves or substantial setup | Approximate/amortize |

The latest diagnostic found about 91–96% near-characteristic solution-error
energy at N128, but only about 1–9% outside the N32 coarse band. Scalar-angular
energy was only about 15–38%. Thus neither radial high-frequency filtering nor
an exclusively scalar output addresses all the observed structure. These are
four development media, one source per medium: they do **not** prove that a
small common response space exists across sources.

The matched characteristic-head pilot reduced mean outer iterations slightly
but was slower than matched CT in 22 of 24 warmed GPU cases. Those timings
were contended. It does not justify more small-head tuning, nor prove that a
larger global architecture will succeed. See the
[completed results](RTE_STRUCTURE_AND_RESOLUTION_RESULTS_2026_09_13.md).

## Concrete architecture: global response-space predictor

The proposed learned map is material -> full-angle basis Z, not residual ->
an unconstrained full solution at every Krylov iteration.

1. **Material encoder.** Input log total/absorption/transport rates, normalized
   physical coordinates, cell size, angular quadrature and phase information.
   Retain physical scales instead of normalizing away optical thickness.
   Use a four-level periodic spatial hierarchy with illustrative widths
   32/64/96/128. Local blocks resolve interfaces; global attention on the
   coarsest tokens communicates across the domain. These widths are starting
   choices, not selected hyperparameters.
2. **Direction-aware communication.** At intermediate/coarse levels, share
   blocks across direction tokens carrying (cos theta, sin theta). Couple
   spatial tokens along discrete characteristic directions and couple angular
   tokens through the known phase kernel plus learned channels. Global
   attention is restricted to coarse tokens, not all N^2 L fine tokens.
   The goal is to infer remote channel/barrier connectivity and angular
   conversion, not merely recognize large/small local coefficients.
3. **Global basis queries.** k=4/8/16 learned query tokens attend to the encoded
   medium. A shared multiscale decoder yields k complex basis fields over
   every retained direction and spatial cell. Use two real output channels;
   preserve the current complex discrete convention rather than silently
   imposing a different Nyquist operator. Translation consistency is tested
   on the predicted projector, since a basis can rotate or change signs.
4. **Physics verification and caching.** Apply the actual discrete A to these
   k fields, rank-filter and normalize their images, then cache the resulting
   basis pair. No full-grid neural inference is needed per outer iteration.
   The setup is invalidated on every change of material or geometry.
5. **Cheap full-angle complement.** Keep HSS/CT on the remaining residual.
   The entire correction range is not truncated to the learned space or to
   scalar moments. Outer FP64 FGMRES checks the unchanged true residual.

This uses successful historical ingredients (cheap collision/streaming
factors, caching, full angular coverage, short-Krylov objectives), while
moving neural capacity to global material dependence. Simply widening the
existing local correction CNN does not implement this architecture.

### Verified augmentation

Write Z in flattened field coordinates. In exact arithmetic, factor

\[
Y=AZ=QR,\qquad W=ZR^{-1},\qquad AW=Q,\quad Q^*Q=I.
\]

Use a rank-revealing factorization; do not invert nearly dependent columns.
The diagnostic uses a thin SVD of Y instead of QR, producing the same image
normalization on the retained range. For a fixed linear base M0, define

\[
M_\theta r=WQ^*r+M_0(I-QQ^*)r.
\]

Then

\[
I-AM_\theta=(I-AM_0)(I-QQ^*).
\]

The represented residual directions are corrected exactly, up to numerical
verification error. This identity does not guarantee fewer GMRES iterations,
a uniformly contractive action, or lower wall time for arbitrary residuals.
The FP32 base is only approximately linear. Keep true-residual monitoring and
a validation-fixed fallback to the base solver for unusable predicted spaces.
Using the image space avoids assuming that A is normal or that Z*AZ is a
well-conditioned reduced matrix. Nonnormal transient behavior is relevant
here; eigenvalues alone are an incomplete training target.

Memory for cached W and Q in complex128 is 32 k L N^2 bytes, excluding work
buffers. At N128,L12,k8 this is 48 MiB; at N256 it is 192 MiB. FP32 caches
halve storage but require a separate accuracy audit. High rank can erase the
benefit through memory traffic even when iteration counts improve.

## Why this might capture genuinely expensive structure

For a scalar/angular-complement split, write the exact operator in blocks.
Eliminating the complementary component produces

\[
S=A_{00}-A_{01}A_{11}^{-1}A_{10}.
\]

The inverse inside this expression contains propagation through the varying
medium and repeated angular interactions. Local diffusion is a useful
approximation in its appropriate regime, not this exact Schur complement.
The learned response space is one way to compress difficult global responses
without applying that nested inverse online.

There may be many independently slow regions. If useful rank grows with the
number of regions or resolution, abandon a single global low-rank basis.
The next representation is a hierarchy of local response spaces with learned
coarse inter-region coupling: a compressed Schur response network. It needs
comparison against adaptive classical coarsening/AMG; a multilevel shape by
itself is not an AI advantage. Ballistic and strongly scattering regimes may
need different compressions; do not assume a scalar diffusive mechanism
explains the present mixed family.

### Hierarchical alternative if a global basis is too restrictive

Use the same global material encoder, but decode response bases per spatial
patch and direction sector at each level. A candidate action has the form

\[
K_\theta(a)r=\sum_\ell U_\ell(a)\,B_\ell(a)\,V_\ell(a)^*r.
\]

Here V restricts into local response coordinates, B communicates between
patches/sectors at that physical scale, and U lifts back to full-angle fields.
Use sparse nearby interactions on fine levels and global communication only
at the coarsest level. Illustrative patch ranks are 4/8, selected by actual
compression error and solve cost. The total rank can grow with the number of
patches; this is not the failed rank-eight global ansatz in disguise.

All coefficient dependence is computed and cached by the nonlinear encoder;
the residual action is linear after setup. Combine K with the full-angle
classical path and recompute the physical defect between corrections. Unlike
the small verified global basis, this representation does not provide AW=Q
for every basis column without expensive setup. It relies on true outer
residual checks and must be tested for adverse nonnormal amplification.

The expensive target is the map from material geometry to inter-patch
response, including multiple scattering paths—not merely local interpolation
weights. Before training, sample off-diagonal response blocks from reference
solves and measure rank versus separation, direction and optical thickness.
If those blocks are not compressible, or an ordinary classical hierarchy
constructs equally good blocks cheaply, this architecture has no established
advantage. This block-compressibility test is the next structural diagnostic
after the bounded global-space test, not an assertion that transport always
has low-rank far-field interactions.

## Training and comparison protocol

First establish the **oracle opportunity**, before a large training run:

- On each development material, collect baseline slow errors from several
  calibration RHS using accurate reference solves. Extract normalized error
  snapshot spaces at ranks 2/4/8/16; also test residual-weighted spaces.
- Evaluate the verified augmentation on different RHS for the same material.
  Include source positions, widths and angular content. No evaluated source
  may contribute to its own basis. Measure rank requirements and the
  conditional benefit before charging teacher construction as offline work.
- Compare simple first-RHS Krylov recycling, then a properly tuned recycling
  method and variable-coefficient DSA/multilevel controls. A teacher-built
  space is expensive classical information, not an AI result.
- Only if the oracle has substantial headroom, train the global encoder on
  separate materials. Distill projectors/principal angles, not vector MSE:
  bases have arbitrary signs and rotations. Fine-tune on short true-residual
  FGMRES trajectories with disjoint RHS. Penalize ineffective rank and setup
  cost; report held-out-material performance, not only held-out sources.

Classical subspace recycling is established, so the proposed contribution
would be *predicting useful spaces before those solves*, not inventing
deflation. See [Parks et al., 2006](https://epubs.siam.org/doi/10.1137/040607277).
Learned coarse/deflation corrections also have precedents in other PDEs,
including [Azulay and Treister, 2022](https://arxiv.org/abs/2203.11025).
These precedents motivate comparisons; they do not establish success for RTE.

### One RHS versus many RHS

Let S be extra neural/verification setup, delta the per-RHS solve-time saving,
and B the number of distinct RHS for a fixed medium. The online break-even
condition is B*delta>S, with delta>0. Count encoder inference, k physical
operator applications, normalization and basis memory traffic. The current
diagnostic uses another k applications to verify AW=Q explicitly and reports
that cost too. Training and teacher generation are separately reported, not
erased from the total resource accounting.

For one RHS and a baseline already requiring only 4–5 iterations, even a good
rank-eight space may cost too much. In that workload, test a source-conditioned
global correction/initial guess once, or a hierarchical response action whose
verification does not require k full solves. It remains subject to true
residual checks and first-solve timing. Repeating one warmed RHS is not a
multiple-source workload and cannot establish amortization.

## Bounded diagnostic completed September 13

`rte/audit_global_response_space.py` tests two existing N64 development media,
eight calibration source translations and six distinct evaluation
translations per medium. HSS8 is the common full-angle base; ranks 2/4/8 are
compared with a first-source eight-step Krylov-span control. It preserves
bases, spectrum, true residuals, iterations, acquisition/setup cost and
single-pass diagnostic timings under
`rte/workstation_runs/global_response_oracle_n64_20260913/`.

This is deliberately a small opportunity check. Translations share source
shape and angular content; materials are not held out. Timings are sequential
single samples, not a performance benchmark. Results must be read with those
limits; an oracle improvement would not yet validate the proposed network.

The refined run in `global_response_oracle_n64_refined_20260913/` additionally
evaluates held-out error/image-space coverage and saves every residual trace.
All bases are frozen before the held-out diagnostic reference solves.

| Method | Mean iterations, 12 RHS | Captured error energy after two HSS8-FGMRES steps | Captured residual energy |
|---|---:|---:|---:|
| HSS8 | 5.0 | — | — |
| Teacher-error rank 2 | 5.0 | 70.3% | 15.0% |
| Teacher-error rank 4 | 5.0 | 73.2% | 20.4% |
| Teacher-error rank 8 | 5.0 | 75.8% | 30.2% |
| First-RHS Krylov rank 8 | 5.0 | 66.1% | 7.9% |

Every method passed the same 1e-5 true-residual tolerance. Single-pass timing
was slower for every augmented variant even before setup, but it is not a
controlled benchmark. The initial run spent about 4.71 seconds per medium
collecting the eight teacher errors, plus roughly 0.10 seconds verifying the
rank-eight basis. That acquisition is not a free AI preconditioner.

**Decision:** no large training run for this global POD space. It captures
substantial solution error but does not eliminate enough relevant residual
directions to reduce outer iterations. Error-energy reconstruction is an
insufficient surrogate objective. The construction is only one teacher-based
POD ansatz, not an optimal rank-eight GMRES preconditioner or a universal
upper bound on learning. Next inspect response-block compression across
physical scales and the more collision-stiff family, retaining strong
classical comparisons. Do not infer success from architecture size alone.

The augmentation identities and rank filtering have dedicated unit tests;
the full RTE test suite passed 96 tests after the implementation.

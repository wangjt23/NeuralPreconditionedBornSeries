# Independent matched reference-defect protocol audit — 2026-10-02

This updates the earlier [four-arm proposal](../reference_control/protocol_audit.md) for the actual prepared ten-arm driver. It is a local static audit before policy freeze, not an execution result. No SSH, GPU work, source export, training, new data or seed9626 access was performed.

## Algebra and state

The implemented pair is the literal original-dtype map

    g = TSA4(r); d = r - A_low(g)
    p_full = g + R0(d); p_q3 = g + R0(Pi3(d)).

[matched_reference.py](matched_reference.py) reuses the same prepared complex64 inverse and original stored material, geometry, phase and quadrature. A segmented prefix is cast back to the input dtype before the defect/addition. Native complex128 promotes the existing inverse rather than rebuilding it. Each output owns storage; guards check tensor identity/version/layout, mode and backend state before replay or native fallback. The compound raw graph captures the entire recipe. No silent precision improvement or altered defect is introduced.

Pi3 retains angular-index modes 0,+1,-1; it restricts the reference INPUT, while its response retains all output angles and spatial frequencies. The positive uniform-weight check supports the stated weighted-projector equivalence. TSA4 and the common N128 reference initialization/tail injection already carry full-angle information. Therefore a negative Q3 result would concern this fixed composition, not every coefficient map in the response range. Both correction arms execute the full 12-angle inverse; Q3 additionally performs angular FFT/ifft. The 17 spatial FFT/ifft calls per compound application versus 31/63 for TSA8/16 are work counts, not wall-time inequalities.

## Screen and acceptance

[matched_screen.py](matched_screen.py) uses frozen development IDs 0/3 at N128/N256. Ten arms comprise the five previous strong implementations, direct-segmented TSA4, and full/Q3 correction with segmented or whole raw capture. Its 252 global entry checks are 4 × (54 action checks + 9 ownership/cache-view bundles), completed before any full solve. Four truncated original-solver prefixes and eight source/prefix geometry targets precede 160 full solves: 40 warmups and 120 measured solves. Proposed limits are 95 s soft, 120 s hard and 12 GiB reserved; external launch/receipt enforcement of hard time and memory remains to be verified.

N128 retains balanced initialization/reference-tail injection; N256 retains zero start. Different algorithms may have different iterations and status. Strong backends, and each full/Q3 backend pair, must match their own identical recipe bitwise and in entire status. Every attempted solve is retained. Eligibility requires finite fields, no solver failure, original-H source-normalized residual ≤1e-5, evaluated discrete relative-error bound ≤1e-3, and paired field difference ≤1e-4. These ordinary FP64 bound evaluations are not interval or continuum certificates.

## Costs and conclusions

The driver bills physical/coercivity setup, the appropriate base, and full constructor costs separately per deployable arm. Segmented compound setup includes the shared TSA4 graph constructor; whole raw setup includes its literal-action construction and capture. Complete solves include initialization, projection, all work and final checks; extra independent acceptance is also timed. Shared benchmark preparation does not make deployment setup free. QA, source I/O and process startup are excluded, so the reported Q1 estimate is setup plus warm accepted latency, not process-cold latency. Constructor order and first-use effects remain relevant with two media per grid.

The proposed policy freezes one backend per correction rule/grid by lowest mean Q1 across both IDs, with deterministic ties. It requires ≥10% mean Q1 and ≥10% mean warm accepted improvement at BOTH grids against the contemporaneous best successful original five, with no individual Q1 regression >10%. All successful ten arms become the future classical denominator. This exploratory classical screen neither relaxes the prior 10% learned goal nor authorizes acquisition/training.

Geometry correctly scores composite p against r and increments against recomputed physical d separately, using the real inner product and captured old search-image span. Its two-proposal fit reports an attained residual, not a full-range optimum. Prefix, homogeneous-matrix construction and geometry costs are diagnostic acquisition: sum the prefix and H0 setup once per context plus each target's total; nested acquisition/fit timings must not be added again. A geometry improvement alone establishes neither solver speed nor affordable coefficient acquisition.

Before freeze, an acquisition-accounting correction was requested: the current N128 balanced reference initialization is built before the prefix timer starts. Record its reference-solve/balance cost separately or include it in prefix acquisition. Full deployable solve timing already includes initialization, so this does not invalidate the fixed-map Q1 accounting.

Two diagnostic refinements were also recommended: promote correction fields before additivity subtraction, and record captured image-basis Gram/orthogonality defects. The current two-pass projection is approximate in floating point. Near-zero/dependent images and any early-converged or unavailable prefix must retain eligibility qualifications. Neither concern changes the deployed maps or supplies an explanation for previous measured runtime gaps.

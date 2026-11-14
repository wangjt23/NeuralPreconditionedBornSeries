# Certified twiddles and primary-source context

This is an independently derived construction for the parent's prospective own radix-two CPU certifier. It neither uses libc sine/cosine accuracy nor claims a new interval FFT method. [INTLAB's primary project page](https://www.tuhh.de/ti3/rump/intlab/) already lists verified forward/backward FFT and real/complex interval arithmetic.

## Rational pi enclosure

For rational0<z<1, form S_n(z)=sum_{k=0}^{n-1}(-1)^k z^{2k+1}/(2k+1). The alternating tail has sign(-1)^n and magnitude at most z^{2n+1}/(2n+1). Therefore the ordered pair S_n and S_n+(-1)^n*z^{2n+1}/(2n+1) encloses atan(z), using exact rational arithmetic.

Let A5=[a5,b5] and A239=[a239,b239] be those enclosures. Machin's identity gives

    pi in [16*a5-4*b239, 16*b5-4*a239].

The reversal in the second subtraction is essential. For completeness, this identity is checked by the tangent addition formula: tan(2atan(1/5))=5/12, tan(4atan(1/5))=120/119, and tan(4atan(1/5)-atan(1/239))=1. The angle lies between0 and pi/2, so it equals pi/4. This algebra and the alternating-series remainder provide the proof; a historical formula citation is not used as a substitute for a remainder bound.

Freeze a rational pi/series precision criterion before any field measurement. Sufficient term counts are determined from the rational remainder bounds, not observed action discrepancies. The authored `certified_twiddles.py` freezes its first-omitted-term target at1e-45 and verifies final rational component widths are strictly below1e-30. These particular sufficient precision choices are mathematical table-generation parameters, not constants inferred from action measurements. The table generator retains integer/rational proof metadata and its final hash.

## Trigonometric enclosures without endpoint-monotonicity pitfalls

Reduce j/k with exact integer quadrant arithmetic to theta=r*pi in [0,pi/2]; use the quadrant identities to restore signs/interchange sine and cosine. Cardinal roots±1 and±i are exact point intervals. Let p=(pi_L+pi_U)/2, theta_c=r*p and d_theta=|r|*(pi_U-pi_L)/2. These are rational.

Evaluate exact rational Taylor polynomials

    S_K(theta_c)=sum_{h=0}^K (-1)^h theta_c^{2h+1}/(2h+1)!,
    C_K(theta_c)=sum_{h=0}^K (-1)^h theta_c^{2h}/(2h)!.

Taylor's theorem, using the bounded derivatives of sin/cos and the vanishing missing-parity term, gives remainder bounds

    B_s=|theta_c|^{2K+3}/(2K+3)!,
    B_c=|theta_c|^{2K+2}/(2K+2)!.

Both sin and cos are globally1-Lipschitz. Thus true sine is in S_K±(B_s+d_theta), and true cosine is in C_K±(B_c+d_theta). This midpoint-plus-Lipschitz route avoids accidentally invoking monotonicity on an enclosure extending past pi/2. Require the rational tail criterion, e.g. B_s,B_c<=2^-160, by a fixed cap/verified rational test. Failure aborts table generation; it does not loosen the criterion.

The preceding Lagrange remainder is a sufficient general construction. The implemented generator reduces to [-pi/4,pi/4], hence its rational midpoint has absolute value below1. Its sine/cosine terms then decrease monotonically in magnitude, so the ordered partial sum and partial-sum-plus-first-omitted-term enclose the value directly. It adds the exact angle radius by the same1-Lipschitz argument. This alternating remainder is valid for the implemented reduction; do not use its monotonicity without that range condition.

Restore the integer quadrant, negate sine for forward exp(-i theta), and conjugate for inverse. Convert every rational endpoint outward to binary64, then **verify its exact represented Fraction against the rational bound**. A preliminary float approximation is only a guess for the endpoint; the exact comparison is the justification. Verify each table pair contains its proven rational enclosing interval and record exact cardinal handling. No rounded pi constant or recursively accumulated twiddle recurrence is an unproved replacement.

Static independent review of the authored generator finds those sign/reduction/remainder/conversion steps consistent: nearest integer quadrant gives |r|<=1/4; negative residual angles reverse the pi endpoints; negative sine and restored quadrants reverse interval endpoints where necessary; forward roots negate the final sine and inverse roots conjugate. Returned buffers are backed by immutable bytes. The twelve authored CPU tests include independent eighth/sixteenth-root algebraic checks, all-stage outward rational containment, exact conjugacy/cardinal cases, cache immutability and blocked-libm construction; an explicitly secondary libc smoke test is not used as a certificate. This reviewer has inspected the source/tests but has not executed them or a numerical project action in this task.

## Primary sources and limits

- [GCC optimization documentation](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html): specifies contraction controls and warns about reassociation, fast-math and rounding-mode-sensitive optimizations. `-frounding-math` alone has a qualified guarantee. Check the actual installed compiler/version and emitted implementation; this page is not a theorem about an uninspected binary.
- [GNU MPFR rounding semantics](https://www.mpfr.org/mpfr-current/mpfr.html#Rounding) and [special functions](https://www.mpfr.org/mpfr-current/mpfr.html#Special-Functions): directed primitives regard represented operands as exact; trigonometric functions support explicit rounding. If a suitable installed version offers sinpi/cospi or rational-turn functions, they can avoid approximate angle formation, but verify the version/API. MPFR is an alternative to the small rational generator, not permission to install a package or a claim about full-FFT cost.
- [INTLAB primary page](https://www.tuhh.de/ti3/rump/intlab/): establishes that verified FFT and interval linear-algebra infrastructure are classical methods. It does not establish our code's inclusion, performance or available licenses/runtime.
- [NumPy1.26 nextafter specification](https://numpy.org/doc/1.26/reference/generated/numpy.nextafter.html): defines the adjacent representable-value operation used for outward endpoint expansion; FFT/BLAS/ordinary reduction are not assumed certified by this documentation.
- [Cooley–Tukey1965, primary AMS publication](https://www.ams.org/journals/mcom/1965-19-090/S0025-5718-1965-0178586-1/S0025-5718-1965-0178586-1.pdf): algorithm lineage for radix factorizations. Primary publication metadata was accessible; direct PDF fetch returned403 in this review. The butterfly inclusion and scaling proofs above are self-contained and do not rely on a fetched implementation or error theorem from that paper.

The native interval and CPU trust assumptions remain explicit. Finite inclusion tests support implementation review but cannot turn an unspecified compiler/library contract into a proof. Once the source-level enclosure theorem and native operation contract are satisfied, no corresponding uniform GPU contract is needed to certify a particular exact saved field's nominal residual.

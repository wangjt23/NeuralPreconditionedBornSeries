# D: collision structure and angular cyclic-block qualification

Actual g=.65 collision is full angular rank12, not isotropic rank1. Exact rational determinants establish rank1536 with the positive128-point scattering profile and a rank>=768 obstruction even after angular-block-diagonal reassignment. The valid general Woodbury system is not reduced to128 unknowns.

The chosen alternative is angular Fourier cyclic block-tridiagonal structure with measured stored-geometry defects, followed by original-G residual repair. It is NOT bitwise exact. Fixed modes[0,1,8] and9 existing vectors all passed original-G target1e-10 after one correction (maximum permitted3); worst repaired residual2.76e-12, dense-action solution difference4.21e-12. Matrix-free originalG matches the dense original block within4.52e-16.

Per-mode setup including assembly fell from0.328-0.350s to0.060-0.092s and stored factor/logical coefficient data from37.75MB to16.27MB. Three-RHS application including repair/check remains about10-11ms, comparable to dense. This is only a CPU small-block qualification, not a complete solver speedup or GPU result. Normal material Fourier profiles were audited and rejected as narrow-band; only the angular direction was implemented.

Read STRUCTURE_REPORT.md for equations, measured defects, costs, caveats and next bounded gate. All source/receipt hashes are in SHA256SUMS. No full17-mode build, full-PDE solve, training or GPU work occurred. A/B/C evidence is unchanged.

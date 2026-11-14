# A: sealed CPU evidence

These are CPU-only measurements. CPU and GPU times must never be mixed in a speed ratio. The paid strong control is compact4 injection with TSA64 for this frozen physical scene and the exact same four source arrays. Original A64 residual<=1e-5, evaluated relative-error bound<=1e-3 and paired field difference<=1e-4 are preserved. First-RHS cost means all setup plus that actual solve/check; it is never the four-RHS total divided by4.

Three fresh, fully paid paired batches: median compact4 total 20.173s versus exact8_inner 17.623s. Paired gains 10.84% to 13.32%. Same-instance only.

Actual first-source costs across the three batches:
- compact4: [7.99076234921813, 7.957475617993623, 7.682190204039216] seconds.
- exact8_inner: [8.522474395111203, 8.39123255177401, 8.539625240955502] seconds.
The exact method is slower on this first-source metric despite its four-source gain.

Read CONFIRMATION_REPORT.md for the timing gate, RESEARCH_REPORT.md for the causal experiments, raw confirmation_cpu_01/ JSON for every field, and SHA256SUMS for the sealed manifest. Reproduce with analyze_results.py and analyze_confirmation.py.

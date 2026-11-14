# Mechanism interpretation, before reading the new candidate result

P8 denotes the orthogonal spatial Fourier projector onto the 17 signed tangential modes; all angular components and normal variation remain. Let S be the already retained search space and Q the orthogonal projector onto A S in the real-stacked inner product. The current residual r is orthogonal to A S up to arithmetic error.

For any proposed coarse correction z, only w = (I - Q) A z supplies a new direction. In exact arithmetic, a one-direction least-squares update reduces squared residual by <r,w>_R^2 / ||w||^2. This is a standard augmentation identity, not a novelty claim.

Even an exact projected solve P8 A z = P8 r does not by itself imply that P8 residual vanishes after the augmented FGMRES update. The update uses the orthogonalized image w and its optimal scalar coefficient, rather than blindly assigning x + z. Projection onto old images can modify its P8 component. Thus the saved P8/complement residual norms are evidence about the actual algorithm, not an assumed exact coarse-projector cancellation.

A second residual-conditioned correction may help because the intervening classical direction changes the residual and its coupling to the retained subspace. It can also fail by returning a nearly redundant direction, producing insufficient reduction to remove a classical call, or costing more than the work removed. The experiment keeps P8 and the factor fixed so that additional mode capacity or new material information cannot explain a gain.

Interpretation is based on accepted original-equation solves, real removed work and fully paid cost. A successful schedule would remain a classical control. It would not establish learned-method usefulness, universal material robustness or GPU throughput. If no classical work is removed, retuning the same schedule is not the next default; the evidence should guide a genuinely different coupling or representation question.

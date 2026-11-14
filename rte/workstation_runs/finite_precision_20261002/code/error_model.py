"""A priori scalar model; backend FFT contract is an explicit free parameter."""
import math
from fractions import Fraction

def exact(value):return Fraction.from_float(float(value))

def ceil_sqrt(q,bits=80):
    q=Fraction(q);assert q>=0
    scale=1<<(2*bits);a=q.numerator*scale;b=q.denominator
    k=math.isqrt(a//b)
    if k*k*b<a:k+=1
    return Fraction(k,1<<bits)

def outward_float(q,upper):
    value=float(q)
    if (exact(value)<q if upper else exact(value)>q):value=math.nextafter(value,math.inf if upper else -math.inf)
    return value

def stored_scales(directions,kx,ky,t,s,phase,weights):
    """Rigorous rational coefficient bounds; no FFT, eigenvalue or field data."""
    directions=[[exact(v)for v in d]for d in directions]
    kx=[exact(v)for v in kx];ky=[exact(v)for v in ky]
    t=[exact(v)for v in t];s=[exact(v)for v in s]
    assert t and all(v>0 for v in t)and all(v>=0 for v in s)
    weights=[exact(v)for v in weights];phase=[[exact(v)for v in row]for row in phase]
    l=len(weights);assert len(directions)==l and len(phase)==l and all(len(row)==l for row in phase)and len(t)==len(s)
    matrix=[[abs(phase[i][j]*weights[j])for j in range(l)]for i in range(l)]
    row=max(sum(v)for v in matrix);col=max(sum(matrix[i][j]for i in range(l))for j in range(l))
    k_bound=ceil_sqrt(row*col)
    d=max(abs(a*x+b*y)for a,b in directions for x in(min(kx),max(kx))for y in(min(ky),max(ky)))
    absolute_d=max(abs(a)*max(map(abs,kx))+abs(b)*max(map(abs,ky))for a,b in directions)
    tm=max(map(abs,t));sm=max(map(abs,s));margin=min(a-b*k_bound for a,b in zip(t,s))
    # Actual physical symbol is assembled64 from the stored geometry. Bound
    # its tiny formation discrepancy instead of calling ideal products exact
    # physical coefficients. Standard binary64 scalar arithmetic is assumed.
    formation64=gamma(2,Fraction(1,1<<53))*absolute_d
    operator=d+formation64+tm+sm*k_bound
    return dict(angular_count=l,scattering_norm_upper=outward_float(k_bound,True),
        scattering_row_sum_upper=outward_float(row,True),scattering_column_sum_upper=outward_float(col,True),
        streaming_norm_upper=outward_float(d+formation64,True),ideal_stored_geometry_streaming_upper=outward_float(d,True),physical_symbol_formation_allowance=outward_float(formation64,True),streaming_absolute_terms_upper=outward_float(absolute_d,True),
        sigma_t_max=outward_float(tm,True),sigma_s_max=outward_float(sm,True),
        operator_norm_upper=outward_float(operator,True),coercivity_lower=outward_float(margin,False),
        coefficient_bound_arithmetic='Exact Fraction of stored binary buffers; rational sqrt rounded upward; final scalar conversion outward',
        validity='Exact nominal stored-coefficient operator only; not a global rounded-backend error guarantee')

def gamma(k,u):
    assert k*u<1
    return k*u/(1-k*u)

def conditional_action_bound(scales,precision,fft_forward,fft_inverse,angular_dot_steps=None):
    """Absolute normwise H error/||v|| under explicit standard-arithmetic contracts.

    Assumes no overflow/underflow/FTZ and real/complex kernels satisfy their
    stated componentwise budgets. Angular dot steps are an assumption unless
    backend execution is established. Neither cuFFT eta is inferred from data.
    """
    assert precision in (32,64)
    assert all(math.isfinite(float(v)) and v>=0 for v in (fft_forward,fft_inverse))
    u=Fraction(1,1<<(24 if precision==32 else 53));l=scales['angular_count']
    k=8*l+4 if angular_dot_steps is None else angular_dot_steps
    assert isinstance(k,int) and k>=0
    d=exact(scales['streaming_norm_upper']);a=exact(scales['streaming_absolute_terms_upper']);tm=exact(scales['sigma_t_max']);sb=exact(scales['sigma_s_max'])*exact(scales['scattering_norm_upper'])
    fft_forward=exact(fft_forward);fft_inverse=exact(fft_inverse)
    # Compare32 to the original physical symbol assembled64. This pays BOTH
    # formation paths; for64 the extra assembly term is conservative.
    delta_d=(gamma(2,u)+(gamma(2,Fraction(1,1<<53))if precision==32 else 0))*a
    fft_product_defect=fft_forward+fft_inverse+u+fft_forward*fft_inverse+u*fft_forward+u*fft_inverse+u*fft_forward*fft_inverse
    streaming=delta_d+(d+delta_d)*fft_product_defect
    collision=u*tm
    scattering=gamma(k,u)*sb
    # Two vector additions: each stored complex component is rounded.
    additions=gamma(2,u)*(d+streaming+tm+collision+sb+scattering)
    return dict(total=outward_float(streaming+collision+scattering+additions,True),streaming=outward_float(streaming,True),
        symbol_formation=outward_float(delta_d,True),collision=outward_float(collision,True),scattering=outward_float(scattering,True),final_additions=outward_float(additions,True),
        unit_roundoff=float(u),angular_dot_steps=k,fft_forward=float(fft_forward),fft_inverse=float(fft_inverse),
        conditional=True,certified_backend_contract=False)

def prospective_budget(scales,field_complex_count=1,tolerance=1e-5,allocation=.01):
    """Tolerance-linked one-percent outer-action budget, set before measurements."""
    m=scales['coercivity_lower'];assert m>0
    tau=Fraction(str(tolerance));fraction=Fraction(str(allocation));mu=Fraction(1,1<<53)
    assert tau>0 and 0<fraction<1 and isinstance(field_complex_count,int) and field_complex_count>0
    nu=gamma(4*int(field_complex_count)+16,mu)
    assert nu<1
    rho=(1-fraction)*tau
    converted_rho=rho*(1+nu)/(1-nu)
    b_allow=exact(m)*(tau-converted_rho-mu)/(1+tau)
    allowed_exact=(b_allow-mu*exact(scales['operator_norm_upper']))/(1+mu)
    allowed=outward_float(allowed_exact,False)
    fixed=conditional_action_bound(scales,64,0.,0.)['total']
    lo,hi=0.,1.
    for _ in range(100):
        mid=(lo+hi)/2
        if conditional_action_bound(scales,64,mid,mid)['total']<=allowed:lo=mid
        else:hi=mid
    return dict(tolerance=tolerance,allocation_fraction=allocation,
        hypothetical_measured_residual_ceiling=outward_float(rho,False),
        assumed_norm_relative_contract=outward_float(nu,True),assumed_subtraction_relative_contract=float(mu),
        field_complex_count=field_complex_count,
        allowed_outer_absolute_error_per_field_norm=allowed,
        assumed_nonfft64_bound=fixed,
        equal_forward_inverse_fft64_contract_required=lo if fixed<=allowed else None,
        derivation='b<=m*(tau-rho_upper-a)/(1+tau); a=mu_sub,b=epsilon_H+mu_sub*(L+epsilon_H),rho_upper=rho_measured*(1+nu_q)/(1-nu_r); unchanged exact tau with all allowances inside fixed1%reserve',
        global_backend_contract_established=False,
        empirical_reference_checks_can_certify_this_contract=False)

def acceptance_upper(measured_residual_upper,field_norm_upper,q_norm_lower,
                     field_norm_lower,margin_lower,action_error_per_norm,
                     subtraction_error_upper):
    values=[measured_residual_upper,field_norm_upper,q_norm_lower,field_norm_lower,margin_lower,action_error_per_norm,subtraction_error_upper]
    assert all(math.isfinite(float(v))and v>=0 for v in values)
    assert q_norm_lower>0 and margin_lower>0 and field_norm_upper>=field_norm_lower
    residual_measure,xhi,qlo,xlo,m,epsilon,sub=map(exact,values)
    error=epsilon*xhi+sub;residual=residual_measure+error;delta=residual/m;den=xlo-delta
    return dict(residual_upper=outward_float(residual,True),relative_residual_upper=outward_float(residual/qlo,True),
        relative_solution_error_upper=outward_float(delta/den,True)if den>0 else None,
        action_plus_subtraction_allowance=outward_float(error,True),conditional_backend_contract=True,
        scalar_bound_arithmetic='Exact Fraction plus outward binary64 conversion; input norm/error bounds remain required')

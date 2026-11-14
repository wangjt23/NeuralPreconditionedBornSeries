"""Execute exact call-site AST with stubs. No torch, arrays or numerical work."""
import ast,json,hashlib
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
old=HERE.parent/'rte_E_N256_20261110_dot_v1/run_n256_gate.py'
new=HERE/'run_permutation_gate.py'
def exercise(path,method,force_bad=False):
    tree=ast.parse(path.read_text());run=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run')
    depth=next(n for n in run.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='depth' for t in n.targets))
    call=next(n for n in run.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='construct')
    guard=next((n for n in run.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='require' and any(isinstance(a,ast.Constant) and a.value=='STOP_METHOD_BINDING' for a in n.value.args)),None)
    recorded=[]
    def construct(b,depth=64):
        recorded.append(depth);actual=64 if force_bad else depth
        # Deliberately keep metadata at requested depth while the real object
        # lies, so the rejection must inspect the object rather than the label.
        return None,None,SimpleNamespace(depth=actual,kind='tsa'),None,{'depth':depth}
    def require(ok,*args):
        if not ok:raise AssertionError(args)
    env=dict(method=method,rhs=[object()],construct=construct,require=require)
    module=ast.Module(body=[depth,call]+([guard] if guard else []),type_ignores=[])
    try:exec(compile(ast.fix_missing_locations(module),str(path),'exec'),env);blocked=False
    except AssertionError:blocked=True
    return dict(requested=16 if method=='tsa16_zero' else 64,passed_argument=recorded[0],guard_present=guard is not None,blocked=blocked)
rows=[]
for method in ['tsa64_zero','compact4','tsa16_zero','E']:
    before=exercise(old,method);after=exercise(new,method)
    assert after['passed_argument']==after['requested'] and not after['blocked'] and after['guard_present']
    if method=='tsa16_zero':assert before['passed_argument']==64 and before['requested']==16
    rows.append(dict(method=method,original=before,repaired=after))
bad=exercise(new,'tsa16_zero',force_bad=True);assert bad['blocked']
result=dict(passed=True,scope='Exact run() depth selection and constructor-call AST executed against stubs; no operator or numerical solver. This is an interface test, not an accuracy qualification.',old_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),methods=rows,deliberate_bad_return=bad)
(HERE/'DEPTH_BINDING_TEST.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

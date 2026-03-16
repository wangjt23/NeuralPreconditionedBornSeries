## Train with preconditioned loss
train_solver.py line 119: USE_PRECOND_RES = True 
## Train with unpreconditioned loss
train_solver.py line 119: USE_PRECOND_RES = False
## inference
    EXP_DIR = 'models/fno_solver_precond_02_19_19_43_50'
inference.py will auto detect whether to use preconditioned loss by checking `EXP_DIR = 'models/fno_solver_precond_02_19_19_43_50'` in line 20
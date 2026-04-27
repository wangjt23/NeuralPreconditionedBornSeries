## Hint
ALL scripts can be seen in `./run.sh`
## run fno solver with direct loss
```bash
python newton.py --model_type FNO_solver \
                 --num_channel_u 48 \
                 --num_layer 4 \
                 --num_channel_f 2    \
                 --final_div_factor 50 \
                 --weight_decay 1e-6 \
                 --lr 1e-4   \
                 --batch_size 50 \
                 --epochs 200 \
                 --loss_type solver
```
## run fno solver with preconditioned loss
```bash
python newton.py --model_type FNO_solver_precond \
                 --num_channel_u 48 \
                 --num_layer 4 \
                 --num_channel_f 2    \
                 --final_div_factor 50 \
                 --weight_decay 1e-6 \
                 --lr 1e-4   \
                 --batch_size 50 \
                 --epochs 200 \
                 --loss_type solver
```
## inference: linearized newton system
```bash
python inference_innerloop.py --model_path "/afs/L202500118/wangjt/work/precond_loss/nolinear/model/FNO_solver_precondns_merge2026-02-20 01:32:13.228650.pt" \
                    --batch_size 1 \
                    --max_iter 100 \
                    --tol 1e-4

```
## inference: newton step
```bash
python inference_newton.py --model_path "/afs/L202500118/wangjt/work/precond_loss/nolinear/model/FNO_solver_precondns_merge2026-02-10 20:26:11.785565.pt" \
                           --sample_index 5001 \
                           --max_newton_iter 10 \
                           --newton_tol 1e-4\
                           --linear_solver_max_iter 100 \
                           --linear_solver_tol 1e-4
```
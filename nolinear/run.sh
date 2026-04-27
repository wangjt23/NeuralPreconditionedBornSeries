# python newton.py --model_type FNO_solver \
#                  --final_div_factor 50 \
#                  --weight_decay 1e-6 \
#                  --lr 1e-4   \
#                  --batch_size 50 \
#                  --epochs 200 \
#                  --loss_type solver

# python newton.py --model_type FNO_solver_precond_bs_l_2 \
#                  --final_div_factor 50 \
#                  --weight_decay 1e-6 \
#                  --lr 1e-4   \
#                  --batch_size 50 \
#                  --epochs 200 \
#                  --loss_type solver

python newton.py --model_type FNO_solver_precond_bs_R_eta \
                 --final_div_factor 50 \
                 --weight_decay 1e-6 \
                 --lr 1e-4   \
                 --batch_size 50 \
                 --epochs 200 \
                 --loss_type solver
# python newton_2.py --model_type MG_solver \
#                  --num_channel_u 48 \
#                  --num_layer 4 \
#                  --num_channel_f 2    \
#                  --final_div_factor 50 \
#                  --weight_decay 1e-6 \
#                  --lr 1e-4   \
#                  --batch_size 50 \
#                  --epochs 500 \
#                  --loss_type solver

# python inference_innerloop.py --model_path "/afs/L202500118/wangjt/work/precond_loss/nolinear/model/FNO_solverns_merge2026-02-10 14:55:20.022450.pt" \
#                     --batch_size 1 \
#                     --max_iter 100 \
#                     --tol 1e-3
# python inference.py --model_path "/home/wangjt/work/precond_loss/Newton/model/FNO_solverns_merge2025-09-14 23:28:10.450609.pt" \
#                     --batch_size 1 \
#                     --max_iter 100 \
#                     --tol 1e-4
# python inference_cmp.py --model_path "/home/wangjt/work/precond_loss/Newton/model/FNO_solverns_merge2025-09-10 16:59:02.562166.pt" \
#                     --batch_size 1 \
#                     --max_iter 1000 \
#                     --tol 1e-4
# python inference_dst.py --log_dir "./dst.log" \
#                     --batch_size 1 \
#                     --max_iter 100 \
#                     --tol 1e-4



# python inference_newton.py --model_path "/afs/L202500118/wangjt/work/precond_loss/nolinear/model/FNO_solverns_merge2026-02-10 20:04:23.369095.pt" \
#                            --sample_index 5025 \
#                            --max_newton_iter 3 \
#                            --newton_tol 1e-4\
#                            --linear_solver_max_iter 100 \
#                            --linear_solver_tol 1e-4




# python inference_innerloop.py --model_path "/afs/L202500118/wangjt/work/precond_loss/nolinear/model/FNO_solver_precondns_merge2026-02-19 21:20:13.597794.pt" \
#                     --batch_size 1 \
#                     --max_iter 100 \
#                     --tol 1e-4

# python inference_innerloop.py --model_path "/afs/L202500118/wangjt/work/precond_loss/nolinear/model/FNO_solver_precondns_merge2026-02-19 21:28:22.942075.pt" \
#                     --batch_size 1 \
#                     --max_iter 100 \
#                     --tol 1e-4

# python inference_newton.py --model_path "/afs/L202500118/wangjt/work/precond_loss/nolinear/model/FNO_solver_precondns_merge2026-02-10 20:26:11.785565.pt" \
#                            --sample_index 5001 \
#                            --max_newton_iter 10 \
#                            --newton_tol 1e-4\
#                            --linear_solver_max_iter 100 \
#                            --linear_solver_tol 1e-4
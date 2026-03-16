
sco acp jobs create \
  --workspace-name=share-space-01e \
  --aec2-name=computing-cluster-01e \
  --job-name="FNO_solver" \
  --container-image-url="registry.cn-sh-01.sensecore.cn/zhicheng_ccr/wjt-ubuntu24.04:wjt-ubuntu-20260311102152" \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=N6lS.Iu.I10.1 \
  --priority=NORMAL \
  --storage-mount=01995896-d54c-7816-90b3-0ce73bce9936:/afs \
  --command="export PATH=/afs/jiajiwei-folder/wangjt/software/miniconda3/bin:$PATH; \
             cd /afs/jiajiwei-folder/wangjt/work/precond_loss/nolinear; \
             nvidia-smi; \
             python newton.py --model_type FNO_solver \
                 --final_div_factor 50 \
                 --weight_decay 1e-6 \
                 --lr 1e-4   \
                 --batch_size 50 \
                 --epochs 200 \
                 --loss_type solver
             "


sco acp jobs create \
  --workspace-name=share-space-01e \
  --aec2-name=computing-cluster-01e \
  --job-name="FNO_solver_precond_bs_l_2" \
  --container-image-url="registry.cn-sh-01.sensecore.cn/zhicheng_ccr/wjt-ubuntu24.04:wjt-ubuntu-20260311102152" \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=N6lS.Iu.I10.1 \
  --priority=NORMAL \
  --storage-mount=01995896-d54c-7816-90b3-0ce73bce9936:/afs \
  --command="export PATH=/afs/jiajiwei-folder/wangjt/software/miniconda3/bin:$PATH; \
             cd /afs/jiajiwei-folder/wangjt/work/precond_loss/nolinear; \
             nvidia-smi; \
             python newton.py --model_type FNO_solver_precond_bs_l_2 \
                              --final_div_factor 50 \
                              --weight_decay 1e-6 \
                              --lr 1e-4   \
                              --batch_size 50 \
                              --epochs 200 \
                              --loss_type solver
            "

sco acp jobs create \
  --workspace-name=share-space-01e \
  --aec2-name=computing-cluster-01e \
  --job-name="FNO_solver_precond_bs_R_eta" \
  --container-image-url="registry.cn-sh-01.sensecore.cn/zhicheng_ccr/wjt-ubuntu24.04:wjt-ubuntu-20260311102152" \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=N6lS.Iu.I10.1 \
  --priority=NORMAL \
  --storage-mount=01995896-d54c-7816-90b3-0ce73bce9936:/afs \
  --command="export PATH=/afs/jiajiwei-folder/wangjt/software/miniconda3/bin:$PATH; \
             cd /afs/jiajiwei-folder/wangjt/work/precond_loss/nolinear; \
             nvidia-smi; \
             python newton.py --model_type FNO_solver_precond_bs_R_eta \
                              --final_div_factor 50 \
                              --weight_decay 1e-6 \
                              --lr 1e-4   \
                              --batch_size 50 \
                              --epochs 200 \
                              --loss_type solver
            "
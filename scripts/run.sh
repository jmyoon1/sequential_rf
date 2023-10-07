TPUNAME='YJM05'

# PROJECTNAME="celeba_32_to_64_i2sb_ncsnpp_noise_0_1"
# python main.py \
#   --config="configs/sr/celeba_32_to_64_i2sb.py" \
#   --eval_folder=$PROJECTNAME \
#   --mode="train" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME \
#   --config.model.high_sigma=0.1


# python main.py \
#   --config="configs/sr/celeba_32_to_64_i2sb.py" \
#   --eval_folder=$PROJECTNAME \
#   --mode="eval" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME \
#   --config.eval.begin_ckpt=1 \
#   --config.eval.end_ckpt=3

# Generate low-resolution + sample
# python main.py \
#   --config='configs/sr/celeba_32_to_64_i2sb.py' \
#   --eval_folder=$PROJECTNAME"_1000" \
#   --mode='eval' \
#   --workdir='exp/'$PROJECTNAME \
#   --log_name=$TPUNAME'-'$PROJECTNAME \
#   --config.eval.begin_ckpt=1 \
#   --config.eval.end_ckpt=1 \
#   --config.low.model.sigma_min=0.1 \
#   --config.eval.sample_low_resolution=True \
#   --config.low.sampling.noise_removal=False \
#   --config.low.run_last_step=False \
#   --config.model.num_scales=1000



# # Generate low-resolution + sample
# python main.py \
#   --config='configs/sr/celeba_32_to_64_i2sb.py' \
#   --eval_folder=$PROJECTNAME"_200_ddim" \
#   --mode='eval' \
#   --workdir='exp/'$PROJECTNAME \
#   --log_name=$TPUNAME'-'$PROJECTNAME \
#   --config.eval.begin_ckpt=1 \
#   --config.eval.end_ckpt=1 \
#   --config.low.model.sigma_min=0.1 \
#   --config.eval.sample_low_resolution=True \
#   --config.low.sampling.noise_removal=False \
#   --config.low.run_last_step=False \
#   --config.sampling.probability_flow=True \
#   --config.model.num_scales=200


# # Generate low-resolution + sample
# python main.py \
#   --config='configs/sr/celeba_32_to_64_i2sb.py' \
#   --eval_folder=$PROJECTNAME"_50_ddim" \
#   --mode='eval' \
#   --workdir='exp/'$PROJECTNAME \
#   --log_name=$TPUNAME'-'$PROJECTNAME \
#   --config.eval.begin_ckpt=1 \
#   --config.eval.end_ckpt=1 \
#   --config.low.model.sigma_min=0.1 \
#   --config.eval.sample_low_resolution=True \
#   --config.low.sampling.noise_removal=False \
#   --config.low.run_last_step=False \
#   --config.sampling.probability_flow=True
# PROJECTNAME="celeba_32_to_64_rf_ncsnpp_noise_0_1"
# python main.py \
#   --config="configs/sr/celeba_32_to_64_rf.py" \
#   --eval_folder=$PROJECTNAME \
#   --mode="train" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME \
#   --config.model.high_sigma=0.1

# python main.py \
#   --config="configs/sr/celeba_32_to_64_rf.py" \
#   --eval_folder=$PROJECTNAME \
#   --mode="train" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME \
#   --config.model.rf_phase=2 \
#   --config.model.high_sigma=0.1 \
#   --config.training.n_iters=1500001


# python main.py \
#   --config='configs/sr/celeba_32_to_64_rf.py' \
#   --eval_folder=$PROJECTNAME"_50_newer" \
#   --mode='eval' \
#   --workdir='exp/'$PROJECTNAME \
#   --log_name=$TPUNAME'-'$PROJECTNAME'-sample-from-train-32-2' \
#   --config.eval.begin_ckpt=4 \
#   --config.eval.end_ckpt=6 \
#   --config.low.model.sigma_min=0.1 \
#   --config.low.sampling.noise_removal=False \
#   --config.low.run_last_step=False \
#   --config.sampling.probability_flow=True \
#   --config.model.num_scales=50


# PROJECTNAME="celeba_48_to_64_rf_ncsnpp"
# python main.py \
#   --config="configs/sr/celeba_48_to_64_rf.py" \
#   --eval_folder=$PROJECTNAME \
#   --mode="train" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME \
#   --config.training.n_iters=500000

# PROJECTNAME="cifar10_example"
# python main.py \
#   --config="configs/rf/cifar10_rf_continuous.py" \
#   --eval_folder=$PROJECTNAME \
#   --mode="train" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME \
#   --config.training.snapshot_statistics=True \
#   --config.training.snapshot_freq=1000 \
#   --config.model.num_scales=100


# Reflow-2
# PROJECTNAME="celeba_64_rf_ncsnpp"
# python main.py \
#   --config="configs/rf/celeba_64_rf_continuous.py" \
#   --eval_folder=$PROJECTNAME \
#   --mode="train" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME"-50000pt-1div-again" \
#   --config.model.rf_phase=2 \
#   --config.training.n_iters=300001 \
#   --config.model.rf_task='gen_t' \
#   --config.training.reflow_source_ckpt=26 \
#   --config.training.reflow_t=1 \
#   --config.training.n_reflow_data=50000 \
#   --config.training.snapshot_statistics=True \
#   --config.training.snapshot_freq=10000

# for i in 100 50 25 10 5 2 1 1000 500 250
# for i in 1000 500
# do
# python main.py \
#   --config="configs/rf/celeba_64_rf_continuous.py" \
#   --eval_folder="rf_solver_"$i \
#   --mode="eval" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME \
#   --config.model.rf_phase=2 \
#   --config.eval.begin_ckpt=1 \
#   --config.eval.end_ckpt=1 \
#   --config.model.num_scales=$i
# done


######### Imagenet-64 ###############

# PROJECTNAME="imagenet_64_rf_ncsnpp"
# python main.py \
#   --config="configs/rf/imagenet_64_rf_continuous.py" \
#   --eval_folder=$PROJECTNAME \
#   --mode="train" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME \
#   --config.model.rf_task='gen' \
#   --config.training.n_iters=1500000


NDIV=6
PROJECTNAME="celeba_64_rf_ncsnpp"
# # Generate reflow data
# python main.py \
#   --config="configs/rf/celeba_64_rf_continuous.py" \
#   --eval_folder=$PROJECTNAME \
#   --mode="train" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME"-50000pt-"$NDIV"div-hard-gen-20000step" \
#   --config.model.rf_phase=2 \
#   --config.training.reflow_source_ckpt=26 \
#   --config.training.reflow_t=$NDIV \
#   --config.training.n_reflow_data=50000 \
#   --config.training.reflow_mode='gen_reflow' \
#   --config.training.soft_division=0.0

# # # train
# python main.py \
#   --config="configs/rf/celeba_64_rf_continuous.py" \
#   --eval_folder=$PROJECTNAME \
#   --mode="train" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME"-50000pt-"$NDIV"div-hard-train-100000step" \
#   --config.model.rf_phase=2 \
#   --config.training.n_iters=100001 \
#   --config.training.reflow_mode='train_reflow' \
#   --config.training.reflow_source_ckpt=26 \
#   --config.training.reflow_t=$NDIV \
#   --config.training.n_reflow_data=50000 \
#   --config.training.snapshot_sampling=True \
#   --config.training.snapshot_statistics=True \
#   --config.training.snapshot_freq=20000 \
#   --config.training.snapshot_save_freq=20000 \
#   --config.model.ema_rate=0.9999 \
#   --config.model.variable_ema_rate=False


# for i in 2 5 10 25 50 100
# do
# python main.py \
#   --config="configs/rf/celeba_64_rf_continuous.py" \
#   --eval_folder="rf_solver_"$NDIV"div_"$i \
#   --mode="eval" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME \
#   --config.model.rf_phase=2 \
#   --config.training.reflow_t=$NDIV \
#   --config.eval.begin_ckpt=2 \
#   --config.eval.end_ckpt=2\
#   --config.model.num_scales=$i \
#   --config.eval.num_samples=50000
# done

# mkdir exp/celeba_6div
# mv exp/celeba_64_rf_ncsnpp/checkpoints-meta_reflow_2 exp/celeba_6div
# mv exp/celeba_64_rf_ncsnpp/checkpoints_reflow_2 exp/celeba_6div
# NDIV=20
# # Generate reflow data
# python main.py \
#   --config="configs/rf/celeba_64_rf_continuous.py" \
#   --eval_folder=$PROJECTNAME \
#   --mode="train" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME"-50000pt-"$NDIV"div-hard-gen-100000step" \
#   --config.model.rf_phase=2 \
#   --config.training.reflow_source_ckpt=26 \
#   --config.training.reflow_t=$NDIV \
#   --config.training.n_reflow_data=50000 \
#   --config.training.reflow_mode='gen_reflow' \
#   --config.training.soft_division=0.0

# # # train
# python main.py \
#   --config="configs/rf/celeba_64_rf_continuous.py" \
#   --eval_folder=$PROJECTNAME \
#   --mode="train" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME"-50000pt-"$NDIV"div-hard-train-100000step" \
#   --config.model.rf_phase=2 \
#   --config.training.n_iters=100001 \
#   --config.training.reflow_mode='train_reflow' \
#   --config.training.reflow_source_ckpt=26 \
#   --config.training.reflow_t=$NDIV \
#   --config.training.n_reflow_data=50000 \
#   --config.training.snapshot_sampling=True \
#   --config.training.snapshot_statistics=True \
#   --config.training.snapshot_freq=20000 \
#   --config.training.snapshot_save_freq=20000 \
#   --config.model.ema_rate=0.9999 \
#   --config.model.variable_ema_rate=False


# mkdir exp/celeba_20div
# mv exp/celeba_64_rf_ncsnpp/checkpoints_reflow_2 exp/cifar10_20div"
# mv exp/celeba_64_rf_ncsnpp/checkpoints-meta_reflow_2 exp/cifar10_20div"

# for NDIV in 20
# do
# PROJECTNAME="celeba_"$NDIV"div"
# for TIMESTEP_STYLE in 'uniform'
# do
# for i in 10 25 50 100
# do
# python main.py \
#   --config="configs/rf/celeba_64_rf_continuous.py" \
#   --eval_folder="rf_solver_"$NDIV"div_"$i"_"$TIMESTEP_STYLE \
#   --mode="eval" \
#   --workdir="exp/"$PROJECTNAME \
#   --log_name=$TPUNAME"-"$PROJECTNAME \
#   --config.model.rf_phase=2 \
#   --config.training.reflow_t=$NDIV \
#   --config.eval.begin_ckpt=1 \
#   --config.eval.end_ckpt=1 \
#   --config.model.num_scales=$i \
#   --config.eval.num_samples=50000 \
#   --config.sampling.timestep_style=$TIMESTEP_STYLE
# done
# done
# done

for NDIV in 20
do
PROJECTNAME="celeba_"$NDIV"div"
for TIMESTEP_STYLE in 'denoising'
do
for i in 1 2 5 10 25 50 100
do
python main.py \
  --config="configs/rf/celeba_64_rf_continuous.py" \
  --eval_folder="rf_solver_"$NDIV"div_"$i"_"$TIMESTEP_STYLE \
  --mode="eval" \
  --workdir="exp/"$PROJECTNAME \
  --log_name=$TPUNAME"-"$PROJECTNAME \
  --config.model.rf_phase=2 \
  --config.training.reflow_t=$NDIV \
  --config.eval.begin_ckpt=1 \
  --config.eval.end_ckpt=1 \
  --config.model.num_scales=$i \
  --config.eval.num_samples=50000 \
  --config.sampling.timestep_style=$TIMESTEP_STYLE
done
done
done



for NDIV in 6
do
PROJECTNAME="celeba_"$NDIV"div"
for TIMESTEP_STYLE in 'denoising'
do
for i in 5 10 25 50 100
do
python main.py \
  --config="configs/rf/celeba_64_rf_continuous.py" \
  --eval_folder="rf_solver_"$NDIV"div_"$i"_"$TIMESTEP_STYLE \
  --mode="eval" \
  --workdir="exp/"$PROJECTNAME \
  --log_name=$TPUNAME"-"$PROJECTNAME \
  --config.model.rf_phase=2 \
  --config.training.reflow_t=$NDIV \
  --config.eval.begin_ckpt=1 \
  --config.eval.end_ckpt=1 \
  --config.model.num_scales=$i \
  --config.eval.num_samples=50000 \
  --config.sampling.timestep_style=$TIMESTEP_STYLE
done
done
done
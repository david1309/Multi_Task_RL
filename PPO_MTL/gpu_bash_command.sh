#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=16000  # memory in Mb
#SBATCH -o gpu_logs/kl_outfile__  # send stdout to sample_experiment_outfile
#SBATCH -e gpu_logs/kl_errfile__  # send stderr to sample_experiment_errfile
#SBATCH -t 48:00:00  # time requested in hour:minute:seconds
#SBATCH -p LongJobs
#SBATCH --nodelist= 19
export CUDA_HOME=/opt/cuda-8.0.44

export CUDNN_HOME=/opt/cuDNN-6.0_8.0

export STUDENT_ID=s1779182

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/
# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mtrl

python train.py BipedalWalker-v2 --task_params 0 --task_name PPO_loss -dcore 64 64 -dhead 64 --pol_loss_type kl --num_episodes 20000 --batch_size 20 --save_rate 500


#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=16000  # memory in Mb
#SBATCH -o sample_experiment_outfile__  # send stdout to sample_experiment_outfile
#SBATCH -e sample_experiment_errfile__  # send stderr to sample_experiment_errfile
#SBATCH -t 72:00:00  # time requested in hour:minute:seconds
#SBATCH -p MSC #LongJobs

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

python train.py BipedalWalker-v2 --task_params 1 2 3 --task_name Wind -dcore 64 32 16 -dhead 128 64 16 --num_episodes 21 --batch_size 3 --save_video False --save_rate 2


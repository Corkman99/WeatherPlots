#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=7
#SBATCH --mem=400G
#SBATCH --output /home/users/f/froelicm/scratch/output/%j/slurm.out
#SBATCH --error /home/users/f/froelicm/scratch/output/%j/slurm.err
#SBATCH --mail-user=marco.froelich@unige.ch
#SBATCH --mail-type=ALL
#SBATCH --time=04:00:00

module load GCC/13.3.0 Python/3.12.3 OpenMPI
 
source $HOME/WeatherPlots/venv/bin/activate

export USE_DASK_DISTRIBUTED=1
export DASK_N_WORKERS=${DASK_N_WORKERS:-6}
export DASK_THREADS_PER_WORKER=${DASK_THREADS_PER_WORKER:-1}
export DASK_MEMORY_LIMIT_PER_WORKER=${DASK_MEMORY_LIMIT_PER_WORKER:-45GB}
export PROGRESS_EVERY=${PROGRESS_EVERY:-2}
export VALID_TIME_CHUNK=${VALID_TIME_CHUNK:-1}
# Number of inits submitted concurrently; defaults to DASK_N_WORKERS
export MAX_CONCURRENT_INITS=${MAX_CONCURRENT_INITS:-$DASK_N_WORKERS}

echo "[$(date '+%F %T')] Starting mean_var_per_forecast_time.py"
echo "[$(date '+%F %T')] Distributed-only mode: USE_DASK_DISTRIBUTED=$USE_DASK_DISTRIBUTED"
echo "[$(date '+%F %T')] DASK_N_WORKERS=$DASK_N_WORKERS DASK_THREADS_PER_WORKER=$DASK_THREADS_PER_WORKER DASK_MEMORY_LIMIT_PER_WORKER=$DASK_MEMORY_LIMIT_PER_WORKER PROGRESS_EVERY=$PROGRESS_EVERY MAX_CONCURRENT_INITS=$MAX_CONCURRENT_INITS VALID_TIME_CHUNK=$VALID_TIME_CHUNK"

srun python $HOME/WeatherPlots/scripts/mean_var_per_forecast_time.py

echo "[$(date '+%F %T')] Finished mean_var_per_forecast_time.py"

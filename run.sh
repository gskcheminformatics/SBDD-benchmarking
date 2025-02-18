#!/bin/bash
#SBATCH --job-name=task3_aggregate
#SBATCH --time=3-00:00:00
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=3
module load mamba
source activate /hpc/mydata/upt/ns833749/sbdd_analysis_env
python run_analysis.py --model_name "Pocket2Mol" --analysis_output_dir "/hpc/scratch/hdd1/ns833749/SBDD_analysis_out_final/" --inference_output_dirs "/hpc/projects/upt/SBDD_benchmarking/inference_auto_task3/Pocket2Mol" --inp_task_files "/hpc/projects/upt/SBDD_benchmarking/Pocket2Mol/task_df_3.csv"

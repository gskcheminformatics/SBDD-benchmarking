from sbdd_bench.sbdd_inference.pocket_ligand_info import getPocketLigandInfo
from sbdd_bench.sbdd_inference.run_sbdd import runSBDDModels
import argparse
import pandas as pd
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--task_num", type=int)
    parser.add_argument("--model_s_dict", type=json.loads)
    parser.add_argument("--num_mols", type=int)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    # Get task DataFrame
    if os.path.isfile(args.data_dir):
        task_df = pd.read_csv(args.data_dir)
    else:
        task_pl_info = getPocketLigandInfo(data_dir=args.data_dir)
        task_df = task_pl_info.task_runs_dataframe()

    # Run required SBDD models
    run_models_init = runSBDDModels(
        task_df=task_df,
        task_num=args.task_num,
        model_s_dict=args.model_s_dict,
        output_dir=args.output_dir,
        num_mols=args.num_mols,
    )
    run_models_init.run()

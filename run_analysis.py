from sbdd_bench.sbdd_analysis import combine_outputs, eval_metrics, task_metrics
from sbdd_bench.sbdd_analysis.constants import TASK2_FUNC_TO_PDBS, TASK3_FUNC_TO_PDBS
from sbdd_bench.sbdd_inference.pocket_ligand_info import BLIND_SET_POCKET_IDS
import pandas as pd
import os
import multiprocessing as mp
import argparse


def run_tasks(
    task_id, pocket_spec_output_df, pocket_id_or_assessment_func, analysis_output_dir
):
    """
    For running tasks outsided the class to avoid pickling errors by multprocessing

    Parameters
    ----------
        task_id: str
            ID of task to be run
        pocket_spec_output_df: pandas DataFrame
            DataFrame object containing PDBs for each blind pocket ID
        pocket_id_or_assessment_func: str
            Blind pocket ID to analyse or string name of function to use for analysis
        analysis_output_dir: str
            Path to write analysis output to

    Returns
    -------
        Saved files
    """

    # Get base metrics for each pocket
    aggregated_scores, per_compound_scores = eval_metrics.metricCalcs(output_df=pocket_spec_output_df).run_metrics()

    if len(per_compound_scores) != 0:
        # Merge per_compound_scores with original dataframe (do not miss any molecules that could not be processed)
        per_compound_scores = pd.merge(
            pocket_spec_output_df,
            per_compound_scores,
            on=["PDB ID", "mol_cond", "mol_true", "mol_pred"],
            how="outer",
        )

        if int(task_id) == 1:
            print("Running task 1 metrics")
            func = getattr(
                task_metrics.task1Metrics(per_compound_scores),
                "prot_" + pocket_id_or_assessment_func,
            )
        elif int(task_id) == 2:
            print("Running task 2 metrics")
            func = getattr(
                task_metrics.task2Metrics(per_compound_scores),
                pocket_id_or_assessment_func,
            )
        elif int(task_id) == 3:
            print("Running task 3 metrics")
            func = getattr(
                task_metrics.task3Metrics(per_compound_scores),
                pocket_id_or_assessment_func,
            )
        try:
            target_spec_metrics = func()
        except Exception as e:
            target_spec_metrics = pd.DataFrame(columns=["PDB ID", "mol_cond", "mol_true", "mol_pred"])
            print("Error:", e)

        # Merge target specific metrics with original dataframe (do not miss any molecules that could not be processed)
        target_spec_metrics = pd.merge(
            pocket_spec_output_df[["PDB ID", "mol_cond", "mol_true", "mol_pred"]],
            target_spec_metrics,
            on=["PDB ID", "mol_cond", "mol_true", "mol_pred"],
            how="outer",
        )

        # Save task metrics
        target_spec_metrics.to_csv(
            os.path.join(
                analysis_output_dir,
                "task{0}_spec_scores_".format(str(task_id))
                + pocket_id_or_assessment_func
                + ".csv",
            ),
            index=False,
        )

    else:
        per_compound_scores.to_csv(
            os.path.join(
                analysis_output_dir,
                "task{0}_per_cmpd_scores_".format(str(task_id))
                + pocket_id_or_assessment_func
                + ".csv",
            ),
            index=False,
        )

    aggregated_scores.to_csv(
        os.path.join(
            analysis_output_dir,
            "task{0}_aggregated_scores_".format(str(task_id))
            + pocket_id_or_assessment_func
            + ".csv",
        ),
        index=False,
    )

def run_task_mp(inference_output_dir, inp_task_file, analysis_output_dir, model_name, subsample=False):
    """
    Runs task metrics with multiprocessing

    Parameters
    ----------
        inference_output_dir: str
            Path to inference output directory
        inp_task_file: str
            Location to task_df_{ID}.csv
        model_name: str
                Pocket2Mol, DiffSBDD, AutoGrow4, or LigBuilderv3
        analysis_output_dir: str
            Path to write analysis output to
        subsample: bool or int
            Boolean False if not sampling 3D structures. Set to int of number of samples wanted for analysis

    Returns
    -------
        Saved files
    """

    
    # Get output df for model
    output_df = combine_outputs.getOutputDf(
        output_dir=inference_output_dir,
        inp_task_file=inp_task_file,
        model_name=model_name, calc_all=False
    ).output_df

    # Read inp_task_file name to understand whether running Task1, Task2, or Task3 metrics
    task_name = os.path.basename(inp_task_file)
    if "1" in task_name:
        pocket_dict = BLIND_SET_POCKET_IDS
        task_id = "1"
    elif "2" in task_name:
        pocket_dict = TASK2_FUNC_TO_PDBS
        task_id = "2"
    elif "3" in task_name:
        pocket_dict = TASK3_FUNC_TO_PDBS
        task_id = "3"
    else:
        raise ValueError("Incorrect task.csv filename")

    iterable = []
    # Generate iterable for correct task type
    for pocket_id_or_func, PDB_list in pocket_dict.items():
        PDB_IDs = [PDB_ID.strip().strip("'") for PDB_ID in PDB_list.split(",")]
        if PDB_IDs[0] == "not 7gax":  # Check for task 3
            pocket_spec_output_df = output_df[~output_df["PDB ID"].isin(["7gax"])]
        else:
            pocket_spec_output_df = output_df[output_df["PDB ID"].isin(PDB_IDs)]
        if len(pocket_spec_output_df) != 0:
            if type(subsample) == int:
                subsample_df = pd.DataFrame()
                for pdb in PDB_IDs:
                    if pdb == "not 7gax":
                        mask = ~pocket_spec_output_df["PDB ID"].isin(["7gax"])
                    else:
                        mask = pocket_spec_output_df["PDB ID"].isin([pdb])
                    
                    if subsample >= len(pocket_spec_output_df[mask]):
                        subsample_df = pd.concat([subsample_df, pd.DataFrame(pocket_spec_output_df[mask])])
                    else:
                        subsample_df = pd.concat([subsample_df, pd.DataFrame(pocket_spec_output_df[mask].sample(subsample))])

                iterable.append((task_id, subsample_df, pocket_id_or_func, analysis_output_dir))
            else:
                iterable.append((task_id, pocket_spec_output_df, pocket_id_or_func, analysis_output_dir))

    # Run iterable with correct function call
    with mp.Pool(42) as pool:
        pool.starmap(run_tasks, iterable)
        pool.close()
        pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name Pocket2Mol, DiffSBDD, LigBuilderv3, or AutoGrow4",
    )
    parser.add_argument(
        "--analysis_output_dir",
        type=str,
        help="Output directory where analysis will be written",
    )
    parser.add_argument(
        "--inference_output_dirs",
        type=str,
        help="Directory/directories where task-specific inference is written for the model",
        nargs="+",
    )
    parser.add_argument(
        "--inp_task_files",
        type=str,
        help="Task-specific file/files used to run the benchmark such as 'task_df_1.csv' 'task_df_2.csv' 'task_df_3.csv'",
        nargs="+",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=False,
        help="Optional number of structures to randomly subsample",
    )

    args = parser.parse_args()

    analysis_output_dir = os.path.join(args.analysis_output_dir, args.model_name)

    # Create output directory if it doesn't exist
    if not os.path.isdir(analysis_output_dir):
        os.makedirs(analysis_output_dir)

    # Run each provided task in list
    for idx in range(len(args.inference_output_dirs)):
        run_task_mp(
            inference_output_dir=args.inference_output_dirs[idx],
            inp_task_file=args.inp_task_files[idx],
            analysis_output_dir=analysis_output_dir,
            model_name=args.model_name,
            subsample=args.subsample
        )

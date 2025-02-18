import yaml
import json
import pandas as pd
import subprocess
import shutil
import os


class runSBDDModels:
    """
    Runs tasks on provided SBDD models

    Parameters
    ----------
        task_df: DataFrame or Path
            Obtained from getPocketInfo containing all PDBs and ligands needed to run. Can also be path to saved csv file
        model_s: nested dictionary
            Containing model names as keys and directories and model checkpoint files as values in format {<model_name>: {'dir': <location>, 'ckpt_file': <ckpt_location>}}
            Allowed model names are "Pocket2Mol", "DiffSBDD", "AutoGrow4", "LigBuilderv3"
            The 'ckpt_file' is not needed for AutoGrow4 or LigBuilderv3
        output_dir: path
            Location to save outputs to
        task_num: int
            Argument indicating the task number for benchmarking
        num_mols: int
            Optional argument for specified number of molecules to be generated across runs. Defaults to 5000
    """

    def __init__(self, task_df, model_s_dict, output_dir, task_num, num_mols=5000):
        if type(task_df) == str:
            self.task_df = pd.read_csv(task_df)
        else:
            self.task_df = task_df

        self.task_num = task_num

        self.model_s_dict = model_s_dict
        self.output_dir = output_dir
        self.num_mols = num_mols

    def create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def set_pocket2mol_yaml(self, yaml_path, num_mols, model_ckpt_file):
        with open(yaml_path, "r") as f:
            params = yaml.safe_load(f)

        params["sample"]["num_samples"] = num_mols
        params["model"]["checkpoint"] = model_ckpt_file

        with open(
            yaml_path.split(".yml")[0] + "_tmp_{0}.yml".format(self.task_num), "w"
        ) as outfile:
            yaml.dump(params, outfile)

    def pocket2mol_python_script(self, pocket2mol_dir):
        """Queues running by bash"""

        if not os.path.isfile(
            os.path.join(pocket2mol_dir, "run_sampling_queuepocket2mol.py")
        ):
            with open(
                os.path.join(pocket2mol_dir, "run_sampling_queuepocket2mol.py"), "w"
            ) as f:
                f.write(
                    """\
import subprocess
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task_df_loc', type=str,default='task_df.csv')
    parser.add_argument('--task_df_idx', type=int)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--config', type=str,default='./sample_for_pdb.yml')
    args = parser.parse_args()
    
    task_df = pd.read_csv(args.task_df_loc)
    sample = task_df.iloc[args.task_df_idx]
    
    com = str(sample['COM_x'])+','+str(sample['COM_y'])+','+str(sample['COM_z'])
    subprocess.call(["python sample_for_pdb.py --pdb_path {0} --center ' {1}' --outdir {2} --config {3}".format(sample['prepped_pdb_file'], com, args.outdir, args.config)], shell=True)"""
                )

    def pocket2mol_bash(self, task_df, output_dir, pocket2mol_dir):
        """Write bash script with one day time limit"""

        # Write Python loop script if not yet created
        self.pocket2mol_python_script(pocket2mol_dir=pocket2mol_dir)

        # Save task df to Pocket2Mol directory
        task_df.to_csv(
            os.path.join(pocket2mol_dir, "task_df_{0}.csv".format(self.task_num)),
            index=False,
        )

        # Set throttle array to run 12 concurrent jobs at a time
        bash = """#!/bin/bash
#SBATCH --job-name=Pocket2Mol_sample
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-{4}%12
#SBATCH --partition=up-gpu
#SBATCH --gres=gpu:1
module load anaconda3
source activate <pocket2mol_env>
cd {0}
mkdir -p slurm_out_job_$SLURM_JOB_ID
python run_sampling_queuepocket2mol.py --task_df_loc {1} --outdir {2}/$SLURM_ARRAY_TASK_ID --config {3} --task_df_idx $SLURM_ARRAY_TASK_ID >> slurm_out_job_$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID""".format(
            pocket2mol_dir,
            os.path.join(pocket2mol_dir, "task_df_{0}.csv".format(self.task_num)),
            output_dir,
            os.path.join(
                pocket2mol_dir,
                "configs/sample_for_pdb_tmp_{0}.yml".format(self.task_num),
            ),
            len(task_df),
        )

        with open(os.path.join(pocket2mol_dir, "sample_task_df.sh"), "w") as f:
            f.write(bash)

    def run_pocket2mol(self, model_ckpt_file):
        """Runs Pocket2Mol SBDD generator. Requires GPU so is queued via bash script. Location of checkpoint file is required"""

        print("-----------Pocket2Mol run starting-----------")

        pocket2mol_dir = self.model_s_dict["Pocket2Mol"]["dir"]
        self.create_dir(directory=os.path.join(self.output_dir, "Pocket2Mol"))

        # Edit the sample_for_pdb.yml file with input parameters if the number of molecules generated needs to be changed
        self.set_pocket2mol_yaml(
            yaml_path=os.path.join(pocket2mol_dir, "configs/sample_for_pdb.yml"),
            num_mols=self.num_mols,
            model_ckpt_file=model_ckpt_file,
        )

        # Write bash script to queue with GPU resource
        self.pocket2mol_bash(
            self.task_df,
            os.path.join(self.output_dir, "Pocket2Mol"),
            pocket2mol_dir=pocket2mol_dir,
        )

        # Queue script on slurm with GPU resource
        subprocess.call(
            ["sbatch", "{}".format(os.path.join(pocket2mol_dir, "sample_task_df.sh"))],
            cwd=pocket2mol_dir,
        )

        print("-----------Pocket2Mol run queued-----------")

    def diffsbdd_python_script(self, diffsbdd_dir):
        """Creates Python script for use in bash array job"""
        if not os.path.isfile(
            os.path.join(diffsbdd_dir, "run_sampling_queue_diffsbdd.py")
        ):
            with open(
                os.path.join(diffsbdd_dir, "run_sampling_queue_diffsbdd.py"), "w"
            ) as f:
                f.write(
                    """\
from lightning_modules import LigandPocketDDPM
from utils import write_sdf_file
import os
from pathlib import Path
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task_df_loc', type=str,default='task_df.csv')
    parser.add_argument('--task_df_idx', type=int)
    parser.add_argument('--model_ckpt_file', type=str)
    parser.add_argument('--num_mols', type=int,default=5000)
    parser.add_argument('--outdir', type=str)
    args = parser.parse_args()
    
    device = 'cpu'
    model = LigandPocketDDPM.load_from_checkpoint(args.model_ckpt_file, map_location=device)
    model = model.to(device)
    
    task_df = pd.read_csv(args.task_df_loc)
    sample = task_df.iloc[args.task_df_idx]
    
    prot = sample["prepped_pdb_file"]
    lig_sdf = sample["ligand_sdf_file"]


    pdb_id_only = sample['PDB ID']
    lig_filename_only = sample['ligand_sdf_file'].split('/')[-1][:-4]

    counter = 0
    # 250 samples at a time
    for i in range(args.num_mols//250):
        counter += 250
        mols = model.generate_ligands(pdb_file=prot, n_samples=250, ref_ligand=lig_sdf)
        sdf_out_filename = lig_filename_only + '_' + str(counter)
        write_sdf_file(Path(args.outdir, f'{sdf_out_filename}_mol.sdf'), mols)
        print(counter, 'molecules generated')"""
                )

    def diffsbdd_bash(self, task_df, num_mols, model_ckpt_file):
        """Queues slurm job array to run DiffSBDD"""

        diffsbdd_dir = self.model_s_dict["DiffSBDD"]["dir"]
        # Create output directory
        output_dir = os.path.join(self.output_dir, "DiffSBDD")
        self.create_dir(directory=output_dir)

        # Write Python script if not yet created
        self.diffsbdd_python_script(diffsbdd_dir=diffsbdd_dir)

        # Save task df to Pocket2Mol directory
        task_df.to_csv(
            os.path.join(diffsbdd_dir, "task_df_{0}.csv".format(self.task_num)),
            index=False,
        )

        # Set throttle array to run 12 concurrent jobs at a time
        bash = """#!/bin/bash
#SBATCH --job-name=DiffSBDD_sample
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --array=0-{4}%12
module load anaconda3
source activate <diffsbdd_env>
cd {0}
mkdir -p slurm_out_job_$SLURM_JOB_ID
python run_sampling_queue_diffsbdd.py --task_df_loc {1} --outdir {2}/$SLURM_ARRAY_TASK_ID --task_df_idx $SLURM_ARRAY_TASK_ID --num_mols {3} --model_ckpt_file {5} >> slurm_out_job_$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID""".format(
            diffsbdd_dir,
            os.path.join(diffsbdd_dir, "task_df_{0}.csv".format(self.task_num)),
            output_dir,
            num_mols,
            len(task_df),
            model_ckpt_file,
        )

        with open(os.path.join(diffsbdd_dir, "sample_task_df.sh"), "w") as f:
            f.write(bash)

    def run_diffsbdd(self, model_ckpt_file):
        """Runs DiffSBDD molecule generator. Location of checkpoint file is required"""

        print("-----------DiffSBDD run starting-----------")

        # Run for each row in task df
        self.diffsbdd_bash(
            task_df=self.task_df,
            num_mols=self.num_mols,
            model_ckpt_file=model_ckpt_file,
        )

        # Queue script on slurm with GPU resource
        subprocess.call(
            [
                "sbatch",
                "{}".format(
                    os.path.join(
                        self.model_s_dict["DiffSBDD"]["dir"], "sample_task_df.sh"
                    )
                ),
            ],
            cwd=self.model_s_dict["DiffSBDD"]["dir"],
        )

        print("-----------DiffSBDD run queued-----------")

    def ag4_python_script(self):
        """Opens json submission script and edits according to task_df"""

        if not os.path.isfile(
            os.path.join(
                self.model_s_dict["AutoGrow4"]["dir"], "run_sampling_queue_ag4.py"
            )
        ):
            with open(
                os.path.join(
                    self.model_s_dict["AutoGrow4"]["dir"], "run_sampling_queue_ag4.py"
                ),
                "w",
            ) as f:
                f.write(
                    """\
import json
import pandas as pd
import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task_df_loc', type=str,default='task_df.csv')
    parser.add_argument('--task_df_idx', type=int)
    parser.add_argument('--num_mols', type=int,default=5000)
    parser.add_argument('--default_json', type=str,default='default_ag4_json.json')
    parser.add_argument('--outdir', type=str)
    args = parser.parse_args()
    
    task_df = pd.read_csv(args.task_df_loc)
    sample = task_df.iloc[args.task_df_idx]
    
    with open(args.default_json) as jsonfile:
        json_inp = json.load(jsonfile)
        
    # Keep all defaults, just need these in json file
    json_inp["filename_of_receptor"] = sample["prepped_pdb_file"]
    json_inp["center_x"] = float(sample["COM_x"])
    json_inp["center_y"] = float(sample["COM_y"])
    json_inp["center_z"] = float(sample["COM_z"])
    json_inp["root_output_folder"] = args.outdir
    json_inp["number_of_mutants"] = 10
    json_inp["number_of_crossovers"] = 10
        
    # Run with json file on terminal
    cmd = ""
    for key, val in json_inp.items():
        cmd += " --" + key + " " + str(val)
    
    # Final command
    # Number of generations kept at 10 for initial run, for the final run use the initial run and generate desired number of molecules through mutations and crossovers
    json_inp["number_of_mutants"] = int(args.num_mols/2)
    json_inp["number_of_crossovers"] = int(args.num_mols/2)
    json_inp["num_generations"] = 11
    
    cmd2 = ""
    for key, val in json_inp.items():
        cmd2 += " --" + key + " " + str(val)
        
    subprocess.call(["python RunAutogrow.py {0} ; python RunAutogrow.py {1}".format(cmd, cmd2)], shell=True)"""
                )

    def ag4_bash(self):
        ag4_dir = self.model_s_dict["AutoGrow4"]["dir"]
        output_dir = os.path.join(self.output_dir, "AutoGrow4")
        # Set throttle array to run 12 concurrent jobs at a time
        bash = """#!/bin/bash
#SBATCH --job-name=AutoGrow4_sample
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --array=0-{0}%12
module load anaconda3
source activate <ag4_env>
cd {1}
mkdir -p slurm_out_job_$SLURM_JOB_ID
python run_sampling_queue_ag4.py --task_df_loc {2} --outdir {3}/$SLURM_ARRAY_TASK_ID --task_df_idx $SLURM_ARRAY_TASK_ID --num_mols {4} >> slurm_out_job_$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID""".format(
            len(self.task_df),
            ag4_dir,
            os.path.join(ag4_dir, "task_df_{0}.csv".format(self.task_num)),
            output_dir,
            self.num_mols,
        )

        with open(
            os.path.join(self.model_s_dict["AutoGrow4"]["dir"], "sample_task_df.sh"),
            "w",
        ) as f:
            f.write(bash)

    def run_ag4(
        self,
        mgltools_dir,
    ):
        """Runs AutoGrow4, MGLTools directory can be changed here"""

        # Save task df to AutoGrow4 directory
        self.task_df.to_csv(
            os.path.join(
                self.model_s_dict["AutoGrow4"]["dir"],
                "task_df_{0}.csv".format(self.task_num),
            ),
            index=False,
        )

        print("-----------AutoGrow4 run starting-----------")
        # Default json - use default options defined in RunAutogrow.py
        if not os.path.isfile(
            os.path.join(self.model_s_dict["AutoGrow4"]["dir"], "default_ag4_json.json")
        ):
            with open(
                os.path.join(
                    self.model_s_dict["AutoGrow4"]["dir"], "default_ag4_json.json"
                ),
                "w",
            ) as json_out:
                default_json = {
                    "size_x": 23,  # Keep the same as Pocket2Mol
                    "size_y": 23,
                    "size_z": 23,
                    "source_compound_file": "source_compounds/Fragment_MW_100_to_150.smi",
                    "mgltools_directory": mgltools_dir,
                    "prepare_receptor4.py": os.path.join(
                        mgltools_dir,
                        "MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py",
                    ),  # The -U flag was edited to remove cleanup steps as proteins were processed outside of AutoGrow
                    "number_of_processors": 4,  # Run on 4 processors
                    "dock_choice": "VinaDocking",
                }

                json.dump(default_json, json_out)

        # Write python script if doesn't exist
        self.ag4_python_script()

        # Submit as bash array job script
        self.ag4_bash()
        subprocess.call(
            [
                "sbatch",
                "{}".format(
                    os.path.join(
                        self.model_s_dict["AutoGrow4"]["dir"], "sample_task_df.sh"
                    )
                ),
            ],
            cwd=self.model_s_dict["AutoGrow4"]["dir"],
        )

        print("-----------AutoGrow4 run queued-----------")

    def write_ligbuilder_cavity_files(self, inp_pdb_file, inp_lig_mol2_file, idx):
        """Writes input cavity files to be submitted to slurm. These are temporary and removed later"""

        ligbuilder_dir = self.model_s_dict["LigBuilderv3"]["dir"]

        # Create if doesn't exist
        self.create_dir(
            directory=os.path.join(ligbuilder_dir, "tmp_inp_{0}".format(self.task_num))
        )

        # Input file to temporary directory
        with open(os.path.join(ligbuilder_dir, "example/cavity.input"), "r") as f:
            lines = f.readlines()
            newlines = []
            for line in lines:
                if "RECEPTOR_FILE" in line:
                    newlines.append(
                        "\t\t\t".join([line.split()[0], inp_pdb_file, "\n"])
                    )
                elif "LIGAND_FILE" in line:
                    newlines.append(
                        "\t\t\t".join([line.split()[0], inp_lig_mol2_file, "\n"])
                    )
                else:
                    newlines.append(line)

            # Add HETMETAL and HETWATER flags to take these into account, will override default parameters
            newlines.append("HETMETAL \t\t\t 1 \n")
            newlines.append("HETWATER \t\t\t 1 \n")

            with open(
                os.path.join(
                    ligbuilder_dir,
                    "tmp_inp_{0}/cavity_{1}.input".format(self.task_num, idx),
                ),
                "w",
            ) as f_out:
                f_out.writelines(newlines)

    def ligbuilder_cavity_bash(self):
        ligbuilder_dir = self.model_s_dict["LigBuilderv3"]["dir"]

        # Set throttle array to run 12 concurrent jobs at a time
        bash = """#!/bin/bash
#SBATCH --job-name=LigBuilderv3_sample
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-{0}%12
module load gcc
cd {1}
mkdir -p slurm_out_job_cavity_$SLURM_JOB_ID
../bin/cavity cavity_$SLURM_ARRAY_TASK_ID.input >> slurm_out_job_cavity_$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID""".format(
            len(self.task_df),
            os.path.join(ligbuilder_dir, "tmp_inp_{0}".format(self.task_num)),
        )

        with open(
            os.path.join(
                self.model_s_dict["LigBuilderv3"]["dir"],
                "tmp_inp_{0}/sample_task_df.sh".format(self.task_num),
            ),
            "w",
        ) as f:
            f.write(bash)

    def write_ligbuilder_build_files(
        self, idx, pocket_atom_file, pocket_grid_file, protein_file
    ):
        """Writes input build files to be submitted to slurm. These are temporary and removed later. NOTE: LigBuilder recommends minimum 10000 molecules to be generated"""

        ligbuilder_dir = self.model_s_dict["LigBuilderv3"]["dir"]

        if self.num_mols < 10000:
            print("LigBuilder recommends a minimum of 10000 molecules to be generated")

        # Input file to temporary directory
        with open(os.path.join(ligbuilder_dir, "build/build.input"), "r") as f:
            lines = f.readlines()
            newlines = []
            for line in lines:
                if "TASK_NAME" in line:
                    newlines.append(
                        "\t\t\t".join([line.split()[0], "build_" + str(idx), "\n"])
                    )
                elif "RESULT_DIR" in line:
                    newlines.append(
                        "\t\t\t".join(
                            [
                                line.split()[0],
                                os.path.join(self.output_dir, "LigBuilderv3", str(idx)),
                                "\n",
                            ]
                        )
                    )
                elif "DESIGN_MODE" in line:
                    newlines.append("\t\t\t".join([line.split()[0], str(0), "\n"]))
                elif "PROTEIN_NUM" in line:
                    newlines.append("\t\t\t".join([line.split()[0], str(1), "\n"]))
                elif "POCKET_ATOM_FILE[1]" in line:
                    newlines.append(
                        "\t\t\t".join([line.split()[0], pocket_atom_file, "\n"])
                    )
                elif "POCKET_GRID_FILE[1]" in line:
                    newlines.append(
                        "\t\t\t".join([line.split()[0], pocket_grid_file, "\n"])
                    )
                elif "PROTEIN_FILE[1]" in line:
                    newlines.append(
                        "\t\t\t".join([line.split()[0], protein_file, "\n"])
                    )
                elif "MOLECULE_NUMBER" in line:
                    newlines.append(
                        "\t\t\t".join([line.split()[0], str(self.num_mols), "\n"])
                    )
                elif "[Growing]MOLECULE_NUMBER" in line:
                    newlines.append(
                        "\t\t\t".join(
                            [line.split()[0], str(int(self.num_mols / 2)), "\n"]
                        )
                    )
                elif "[Linking]MOLECULE_NUMBER" in line:
                    newlines.append(
                        "\t\t\t".join(
                            [line.split()[0], str(int(self.num_mols / 4)), "\n"]
                        )
                    )
                elif "INCLUDE" in line:
                    newlines.append(
                        "\t\t\t".join(
                            [line.split()[0], "../default/usersettings.input", "\n"]
                        )
                    )
                else:
                    newlines.append(line)
            newlines.append("\t\t\t".join(["MINIMAL_PKD[]", "2", "\n"]))
            newlines.append("\t\t\t".join(["MINIMAL_AVER_PKD[]", "0", "\n"]))

            with open(
                os.path.join(
                    ligbuilder_dir,
                    "tmp_inp_{0}/build_{1}.input".format(self.task_num, idx),
                ),
                "w",
            ) as f_out:
                f_out.writelines(newlines)

    def ligbuilder_python_script(self):
        """Queues running through bash"""

        ligbuilder_dir = self.model_s_dict["LigBuilderv3"]["dir"]

        if not os.path.isfile(
            os.path.join(ligbuilder_dir, "run_build_step_ligbuilder.py")
        ):
            with open(
                os.path.join(ligbuilder_dir, "run_build_step_ligbuilder.py"), "w"
            ) as f:
                f.write(
                    """\
import subprocess
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int)
    args = parser.parse_args()
    
    proc = subprocess.Popen(["../bin/build -Automatic build_{0}.input".format(args.idx)], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(300) # Wait 5 minutes before starting build
    subprocess.call(["chmod +x run_build_{0}.list ; ./run_build_{0}.list".format(args.idx)], shell=True)"""
                )

    def ligbuilder_build_step_bash(self):
        ligbuilder_dir = self.model_s_dict["LigBuilderv3"]["dir"]

        # Set throttle array to run 12 concurrent jobs at a time
        bash = """#!/bin/bash
#SBATCH --job-name=LigBuilderv3_sample # Same name given to queue after cavity has completed
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --array=0-{0}%12
#SBATCH --dependency=singleton
module load gcc
cd {1}
mkdir -p slurm_out_job_build_$SLURM_JOB_ID
module load anaconda3
python ../run_build_step_ligbuilder.py --idx $SLURM_ARRAY_TASK_ID >> slurm_out_job_build_$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID
""".format(
            len(self.task_df),
            os.path.join(ligbuilder_dir, "tmp_inp_{0}".format(self.task_num)),
        )

        with open(
            os.path.join(
                self.model_s_dict["LigBuilderv3"]["dir"],
                "tmp_inp_{0}/sample_task_df_2.sh".format(self.task_num),
            ),
            "w",
        ) as f:
            f.write(bash)

    def run_ligbuilder(self):
        """Runs LigBuilderv3"""

        print("-----------LigBuilderv3 run starting-----------")

        ligbuilder_dir = self.model_s_dict["LigBuilderv3"]["dir"]

        # Create output directory
        self.create_dir(directory=os.path.join(self.output_dir, "LigBuilderv3"))

        # Loop through each run
        # Copy PDB and mol2 files to new output directory where LigBuilder outputs will be written to
        # e.g. cavity outputs will be stored in <location_to_pdb_file>/<pdb_file_name>_pocket_1.txt and <location_to_pdb_file>/<pdb_file_name>_grid_1.txt
        for idx, row in self.task_df.iterrows():
            pdb_file = row["prepped_pdb_file"]
            mol2_file = row["ligand_sdf_file"][:-4] + ".mol2"

            run_dir = os.path.join(self.output_dir, "LigBuilderv3", str(idx))
            self.create_dir(directory=run_dir)

            shutil.copy(pdb_file, run_dir)
            shutil.copy(mol2_file, run_dir)

            # Write cavity files to tmp_inp
            new_pdb_file_loc = os.path.join(run_dir, pdb_file.split("/")[-1])
            new_mol2_file_loc = os.path.join(run_dir, mol2_file.split("/")[-1])

            self.write_ligbuilder_cavity_files(
                inp_pdb_file=new_pdb_file_loc,
                inp_lig_mol2_file=new_mol2_file_loc,
                idx=idx,
            )
            self.write_ligbuilder_build_files(
                idx=idx,
                pocket_atom_file=new_pdb_file_loc.split(".pdb")[0] + "_pocket_1.txt",
                pocket_grid_file=new_pdb_file_loc.split(".pdb")[0] + "_grid_1.txt",
                protein_file=new_pdb_file_loc,
            )

        # Write Python script
        self.ligbuilder_python_script()
        # Create appropriate bash scripts
        self.ligbuilder_cavity_bash()
        self.ligbuilder_build_step_bash()

        subprocess.call(
            [
                "sbatch",
                "{}".format(
                    os.path.join(
                        self.model_s_dict["LigBuilderv3"]["dir"],
                        "tmp_inp_{0}/sample_task_df.sh".format(self.task_num),
                    )
                ),
            ],
            cwd=os.path.join(
                self.model_s_dict["LigBuilderv3"]["dir"],
                "tmp_inp_{0}".format(self.task_num),
            ),
        )
        print("-----------LigBuilderv3 cavity mode queued-----------")

        subprocess.call(
            [
                "sbatch",
                "{}".format(
                    os.path.join(
                        self.model_s_dict["LigBuilderv3"]["dir"],
                        "tmp_inp_{0}/sample_task_df_2.sh".format(self.task_num),
                    )
                ),
            ],
            cwd=os.path.join(
                self.model_s_dict["LigBuilderv3"]["dir"],
                "tmp_inp_{0}".format(self.task_num),
            ),
        )
        print("-----------LigBuilderv3 build mode queued-----------")

    def run(self):
        for model_name in list(self.model_s_dict.keys()):
            if model_name == "Pocket2Mol":
                self.run_pocket2mol(
                    model_ckpt_file=self.model_s_dict["Pocket2Mol"]["ckpt_file"]
                )
            elif model_name == "DiffSBDD":
                self.run_diffsbdd(
                    model_ckpt_file=self.model_s_dict["DiffSBDD"]["ckpt_file"]
                )
            elif model_name == "AutoGrow4":
                self.run_ag4()
            elif model_name == "LigBuilderv3":
                self.run_ligbuilder()
            else:
                print("Model not found")

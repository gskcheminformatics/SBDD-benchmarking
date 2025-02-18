import pandas as pd
import os
import gzip
import os
import json
import random
import string
from rdkit import RDLogger, Chem

RDLogger.DisableLog("rdApp.*")


class getOutputDf:
    def __init__(self, output_dir, inp_task_file, model_name, calc_all=True):
        """
        Creates a joined DataFrame containing the input task DataFrame used for the run and the files generated in the output directory

        Parameters
        ----------
            output_dir:  str
                Directory where mol gen outputs are stored
            inp_task_file: str
                Input csv containing PDB ID and locations of input PDB and SDF files with columns PDB ID,prepped_pdb_file,ligand_sdf_file,COM_x,COM_y,COM_z
            model_name: str
                Pocket2Mol, DiffSBDD, AutoGrow4, or LigBuilderV3

        Returns
        -------
            output_df: DataFrame containing input task values and locations to output files
        """

        self.out_dir = output_dir
        self.task_df = pd.read_csv(inp_task_file)
        self.model_name = model_name

        self.output_df = self.df_from_model_dir(calc_all=calc_all)

    def extract_p2m_outputs(self):
        """
        Extracts all outputs from Pocket2Mol directory

        Returns
        -------
            task_df_with_outputs: DataFrame with locations to output files and input files
        """

        # For each target (directory), extract SDF samples
        target_indices = [
            i
            for i in os.listdir(self.out_dir)
            if os.path.isdir(os.path.join(self.out_dir, i))
        ]

        task_df_with_outputs = pd.DataFrame()

        PDBs = []
        prepped_files = []
        ligand_files = []
        COM_x = []
        COM_y = []
        COM_z = []
        sdf_path = []

        # For each target index, extract directory then SDF files
        for ind in target_indices:
            ind = str(ind)
            sample_dirs = os.listdir(os.path.join(self.out_dir, ind))
            # Select most recent sample
            sample_dirs = sample_dirs[-1]

            # Connect indices with input PDB-ligand files using log.txt files
            log_file = os.path.join(self.out_dir, ind, sample_dirs, "log.txt")

            try:
                with open(log_file, "r") as f:
                    info_line = f.readline()

                    log_COM_x = (
                        info_line.split("center=")[-1]
                        .split(",")[0]
                        .strip("[]")
                        .strip(" ")
                    )
                    log_COM_y = (
                        info_line.split("center=")[-1]
                        .split(",")[1]
                        .strip("[]")
                        .strip(" ")
                    )
                    log_COM_z = (
                        info_line.split("center=")[-1]
                        .split(",")[2]
                        .strip("[]")
                        .strip(" ")
                    )

                    inp_pdb_name = (
                        info_line.split("pdb_path=")[-1]
                        .split("/")[-1]
                        .strip("\n")
                        .strip("()")
                        .strip("''")
                    )

                    # Get the correct PDB ID
                    row_from_df = self.task_df[
                        self.task_df["prepped_pdb_file"].str.contains(inp_pdb_name)
                    ]

                    if len(row_from_df) > 1:
                        # Get the correct input ligand file based on COM_x, COM_y, and COM_z
                        row = self.task_df[
                            (
                                self.task_df["prepped_pdb_file"].str.contains(
                                    inp_pdb_name
                                )
                            )
                            & (self.task_df["COM_x"] == float(log_COM_x))
                            & (self.task_df["COM_y"] == float(log_COM_y))
                            & (self.task_df["COM_z"] == float(log_COM_z))
                        ]
                        row = row.iloc[0, :]
                    else:
                        row = row_from_df
                        row = row.iloc[0, :]

                    try:
                        sdf_files = [
                            os.path.join(self.out_dir, ind, sample_dirs, "SDF", i)
                            for i in os.listdir(
                                os.path.join(self.out_dir, ind, sample_dirs, "SDF")
                            )
                        ]
                        if len(sdf_files) != 0:
                            for sdf_file in sdf_files:
                                PDBs.append(row["PDB ID"])
                                prepped_files.append(row["prepped_pdb_file"])
                                ligand_files.append(row["ligand_sdf_file"])
                                COM_x.append(row["COM_x"])
                                COM_y.append(row["COM_y"])
                                COM_z.append(row["COM_z"])
                                sdf_path.append(
                                    str(
                                        os.path.join(
                                            self.out_dir,
                                            ind,
                                            sample_dirs,
                                            "SDF",
                                            sdf_file,
                                        )
                                    )
                                )
                        else:
                            PDBs.append(row["PDB ID"])
                            prepped_files.append(row["prepped_pdb_file"])
                            ligand_files.append(row["ligand_sdf_file"])
                            COM_x.append(row["COM_x"])
                            COM_y.append(row["COM_y"])
                            COM_z.append(row["COM_z"])
                            sdf_path.append("No SDF files generated")

                    except Exception as e:
                        PDBs.append(row["PDB ID"])
                        prepped_files.append(row["prepped_pdb_file"])
                        ligand_files.append(row["ligand_sdf_file"])
                        COM_x.append(row["COM_x"])
                        COM_y.append(row["COM_y"])
                        COM_z.append(row["COM_z"])
                        sdf_path.append(e)

            except Exception as e:
                PDBs.append(row["PDB ID"])
                prepped_files.append(row["prepped_pdb_file"])
                ligand_files.append(row["ligand_sdf_file"])
                COM_x.append(row["COM_x"])
                COM_y.append(row["COM_y"])
                COM_z.append(row["COM_z"])
                sdf_path.append(e)

        task_df_with_outputs["PDB ID"] = PDBs
        task_df_with_outputs["mol_cond"] = prepped_files
        task_df_with_outputs["mol_true"] = ligand_files
        task_df_with_outputs["COM_x"] = COM_x
        task_df_with_outputs["COM_y"] = COM_y
        task_df_with_outputs["COM_z"] = COM_z
        task_df_with_outputs["mol_pred"] = sdf_path

        return task_df_with_outputs

    def extract_diffsbdd_outputs(self, calc_all=False):
        """
        Extracts all outputs from DiffSBDD directory

        Returns
        -------
            task_df_with_outputs: DataFrame with locations to output files and input files
        """

        task_df_with_outputs = pd.DataFrame()

        PDBs = []
        prepped_files = []
        ligand_files = []
        COM_x = []
        COM_y = []
        COM_z = []
        sdf_path = []

        target_indices = [
            i
            for i in os.listdir(self.out_dir)
            if os.path.isdir(os.path.join(self.out_dir, i))
            and "tmp_out" not in i
        ]

        for idx in target_indices:
            output_dir = os.path.join(self.out_dir, str(idx))

            # Separate ligands into their own sdf files per output
            tmp_output_dir = os.path.join(output_dir, "tmp_outputs")

            if calc_all == True or not os.path.exists(tmp_output_dir):
                if not os.path.exists(tmp_output_dir):
                    os.makedirs(tmp_output_dir)

            # Get all files in output directory
            all_output_files = [i for i in os.listdir(output_dir)]

            # Extract PDB file and SDF file from filename (if mol name in file exists - important for Task1)
            if (
                ".pdb" in all_output_files[0]
                and all_output_files[0].split(".pdb")[-1].split("_")[2] == "lig"
            ):
                # Extract PDB and ligand filenames
                pdb_filename = all_output_files[0].split(".pdb")[0]
                lig_filename_contains = "_".join(
                    all_output_files[0].split(".pdb")[-1].split("_")[1:-2]
                )

                # Get the correct row
                row_from_df = self.task_df[
                    (self.task_df["prepped_pdb_file"].str.contains(pdb_filename))
                    & (
                        self.task_df["ligand_sdf_file"].str.contains(
                            lig_filename_contains
                        )
                    )
                ]
                #row = row_from_df.iloc[0, :]

            else:
                # Extract PDB filename and select ligand file from task df
                lig_filename_contains = "_".join(all_output_files[0].split("_")[0:-2])
                row_from_df = self.task_df[
                    self.task_df["ligand_sdf_file"].str.contains(lig_filename_contains)
                ]
                #row = row_from_df.iloc[0, :]

            try:
                row = row_from_df.iloc[0, :]

                protein_file = row["prepped_pdb_file"]
                lig_file = row["ligand_sdf_file"]

                # If molecules were generated for this PDB example, otherwise the directory will not exist
                try:    
                    ligand_sdf_files = [
                        os.path.join(self.out_dir, str(idx), file)
                        for file in os.listdir(os.path.join(self.out_dir, str(idx))) if ".sdf" in file
                    ]
                except Exception as e:
                    PDBs.append(row["PDB ID"])
                    prepped_files.append(row["prepped_pdb_file"])
                    ligand_files.append(row["ligand_sdf_file"])
                    COM_x.append(row["COM_x"])
                    COM_y.append(row["COM_y"])
                    COM_z.append(row["COM_z"])
                    sdf_path.append(e)
                    ligand_sdf_files = []

                if calc_all == True:
                    # For each sdf file, create a separate tmp directory of all sdf predictions
                    for file in ligand_sdf_files:
                        try:
                            suppl = Chem.SDMolSupplier(file)
                        except:
                            suppl = None

                        if suppl is not None:
                            mol_counter = 0
                            file_num = file.split("/")[-1].split("_lig_")[-1].split(".sdf")[0]
                            for mol in suppl:
                                if mol is not None:
                                    output_sdf_file = (
                                        protein_file.split("/")[-1].split(".pdb")[0]
                                        + "_"
                                        + lig_file.split("/")[-1].split(".sdf")[0]
                                        + "_"
                                        + file_num
                                        + "_"
                                        + str(mol_counter)
                                        + ".sdf"
                                    )
                                    mol_counter += 1
                                    writer = Chem.SDWriter(
                                        os.path.join(tmp_output_dir, output_sdf_file)
                                    )
                                    writer.write(mol)

                                    PDBs.append(row["PDB ID"])
                                    prepped_files.append(row["prepped_pdb_file"])
                                    ligand_files.append(row["ligand_sdf_file"])
                                    COM_x.append(row["COM_x"])
                                    COM_y.append(row["COM_y"])
                                    COM_z.append(row["COM_z"])
                                    sdf_path.append(
                                        str(os.path.join(tmp_output_dir, output_sdf_file))
                                    )
                else:
                    out_lig_sdf_files = [
                        os.path.join(tmp_output_dir, i) for i in os.listdir(tmp_output_dir)
                    ]

                    if len(out_lig_sdf_files) != 0:
                        for sdf_file in out_lig_sdf_files:
                            PDBs.append(row["PDB ID"])
                            prepped_files.append(row["prepped_pdb_file"])
                            ligand_files.append(row["ligand_sdf_file"])
                            COM_x.append(row["COM_x"])
                            COM_y.append(row["COM_y"])
                            COM_z.append(row["COM_z"])
                            sdf_path.append(str(sdf_file))
                    else:
                        PDBs.append(row["PDB ID"])
                        prepped_files.append(row["prepped_pdb_file"])
                        ligand_files.append(row["ligand_sdf_file"])
                        COM_x.append(row["COM_x"])
                        COM_y.append(row["COM_y"])
                        COM_z.append(row["COM_z"])
                        sdf_path.append("Errno")
            
            except Exception as e:
                print(e)

        task_df_with_outputs["PDB ID"] = PDBs
        task_df_with_outputs["mol_cond"] = prepped_files
        task_df_with_outputs["mol_true"] = ligand_files
        task_df_with_outputs["COM_x"] = COM_x
        task_df_with_outputs["COM_y"] = COM_y
        task_df_with_outputs["COM_z"] = COM_z
        task_df_with_outputs["mol_pred"] = sdf_path

        return task_df_with_outputs

    def extract_autogrow4_outputs(self, calc_all=False):
        """Extracts all outputs from AutoGrow4 directory"""

        task_df_with_outputs = pd.DataFrame()

        PDBs = []
        prepped_files = []
        ligand_files = []
        COM_x = []
        COM_y = []
        COM_z = []
        sdf_path = []

        target_indices = [
            i
            for i in os.listdir(self.out_dir)
            if os.path.isdir(os.path.join(self.out_dir, i))
        ]

        for idx in target_indices:
            output_dir = os.path.join(self.out_dir, str(idx))

            # Get vars.json
            json_file = os.path.join(output_dir, "Run_0", "vars.json")

            try:
                with open(json_file, "r") as f:
                    info = json.load(f)

                    inp_pdb_name = info["filename_of_receptor"]
                    log_COM_x = info["center_x"]
                    log_COM_y = info["center_y"]
                    log_COM_z = info["center_z"]

                    # Get the correct PDB ID
                    row_from_df = self.task_df[
                        self.task_df["prepped_pdb_file"].str.contains(inp_pdb_name)
                    ]

                    if len(row_from_df) > 1:
                        # Get the correct input ligand file based on COM_x, COM_y, and COM_z
                        row = self.task_df[
                            (
                                self.task_df["prepped_pdb_file"].str.contains(
                                    inp_pdb_name
                                )
                            )
                            & (self.task_df["COM_x"] == float(log_COM_x))
                            & (self.task_df["COM_y"] == float(log_COM_y))
                            & (self.task_df["COM_z"] == float(log_COM_z))
                        ]
                        row = row.iloc[0, :]
                    else:
                        row = row_from_df
                        row = row.iloc[0, :]

                    # If it doesn't read generation_11 where 10,000 molecules need to be sampled, combine the last two generations
                    all_generations = [
                        i
                        for i in os.listdir(os.path.join(output_dir, "Run_0"))
                        if "generation" in i.lower() and "failed" not in i.lower()
                    ]
                    all_generations = sorted(
                        all_generations, key=lambda x: int(x.split("_")[-1])
                    )

                    if "generation_11" not in all_generations:
                        sample_gens = all_generations[-2:]
                    else:
                        sample_gens = ["generation_11"]

                    # If it cannot find sampled PDBs for generation 11, take last two generations again, otherwise, skip
                    try:
                        sample_gens = [
                            [
                                os.path.join(output_dir, "Run_0", i, "PDBs", j)
                                for j in os.listdir(
                                    os.path.join(output_dir, "Run_0", i, "PDBs")
                                )
                            ]
                            for i in sample_gens
                        ]
                    except:
                        if sample_gens[0] == "generation_11":
                            all_generations_without_11 = all_generations[:-1]
                            sample_gens = all_generations_without_11[-2:]
                            sample_gens = [
                                [
                                    os.path.join(output_dir, "Run_0", i, "PDBs", j)
                                    for j in os.listdir(
                                        os.path.join(output_dir, "Run_0", i, "PDBs")
                                    )
                                ]
                                for i in sample_gens
                            ]
                        else:
                            sample_gens = []
                            continue

                    sample_gens = [j for i in sample_gens for j in i]

                    # Get compressed_PDBs.txt.gz
                    for compressed_PDB in sample_gens:
                        # Separate ligands into their own sdf files per output
                        tmp_output_dir = os.path.join(output_dir, "tmp_outputs")

                        # Only generate outputs if tmp dir does not exist
                        if calc_all == True or not os.path.exists(tmp_output_dir):
                            if not os.path.exists(tmp_output_dir):
                                os.makedirs(tmp_output_dir)

                            with gzip.open(compressed_PDB, "rb") as f_in:
                                readlines = f_in.readlines()

                                start_pdb = False
                                pdb_string = []

                                mol_counter = 0
                                for line in readlines:
                                    line = line.decode("ascii")
                                    if (
                                        "File_name" in line
                                        and ".pdb" in line.strip("\n")[-5:]
                                    ):
                                        start_pdb = True

                                    if start_pdb:
                                        pdb_string.append(line)

                                    if (
                                        start_pdb
                                        and ".pdb" in line.strip("\n")[-5:]
                                        and "END_FILE" in line
                                    ):
                                        start_pdb = False

                                        if len(pdb_string) != 0:
                                            mol_counter += 1
                                            with open(
                                                "tmp_pdb.pdb", "w"
                                            ) as out_pdb_file:
                                                out_pdb_file.write(
                                                    "".join(pdb_string[1:-1])
                                                )

                                            mol = Chem.MolFromPDBFile(
                                                "tmp_pdb.pdb", removeHs=False
                                            )

                                            if mol is not None:
                                                writer = Chem.SDWriter(
                                                    os.path.join(
                                                        tmp_output_dir,
                                                        str(mol_counter) + ".sdf",
                                                    )
                                                )
                                                writer.write(mol)

                                            os.remove("tmp_pdb.pdb")

                                        pdb_string = []

                    if len(sample_gens) != 0:
                        ligand_sdf_files = [
                            os.path.join(tmp_output_dir, i)
                            for i in os.listdir(tmp_output_dir)
                        ]

                        for sdf_file in ligand_sdf_files:
                            PDBs.append(row["PDB ID"])
                            prepped_files.append(row["prepped_pdb_file"])
                            ligand_files.append(row["ligand_sdf_file"])
                            COM_x.append(row["COM_x"])
                            COM_y.append(row["COM_y"])
                            COM_z.append(row["COM_z"])
                            sdf_path.append(str(sdf_file))
                    else:
                        PDBs.append(row["PDB ID"])
                        prepped_files.append(row["prepped_pdb_file"])
                        ligand_files.append(row["ligand_sdf_file"])
                        COM_x.append(row["COM_x"])
                        COM_y.append(row["COM_y"])
                        COM_z.append(row["COM_z"])
                        sdf_path.append("Errno")

            except Exception as e:
                PDBs.append(row["PDB ID"])
                prepped_files.append(row["prepped_pdb_file"])
                ligand_files.append(row["ligand_sdf_file"])
                COM_x.append(row["COM_x"])
                COM_y.append(row["COM_y"])
                COM_z.append(row["COM_z"])
                sdf_path.append("Errno")

        task_df_with_outputs["PDB ID"] = PDBs
        task_df_with_outputs["mol_cond"] = prepped_files
        task_df_with_outputs["mol_true"] = ligand_files
        task_df_with_outputs["COM_x"] = COM_x
        task_df_with_outputs["COM_y"] = COM_y
        task_df_with_outputs["COM_z"] = COM_z
        task_df_with_outputs["mol_pred"] = sdf_path

        return task_df_with_outputs

    def extract_ligbuilder_outputs(self, calc_all=False):
        """Extracts all outputs from LigBuilderV3 directory"""

        task_df_with_outputs = pd.DataFrame()

        target_indices = [
            i
            for i in os.listdir(self.out_dir)
            if os.path.isdir(os.path.join(self.out_dir, i))
        ]

        PDBs = []
        prepped_files = []
        ligand_files = []
        COM_x = []
        COM_y = []
        COM_z = []
        sdf_path = []

        for idx in target_indices:
            out_dir = os.path.join(self.out_dir, str(idx), "output_build_" + str(idx))

            # Get input PDB file
            inp_prot_filename = [
                i[:-4]
                for i in os.listdir(os.path.join(self.out_dir, str(idx)))
                if (i.endswith("prepped.pdb") or i.endswith("protein.pdb"))
            ][0]
            # Get input ligand file
            inp_lig_filename = [
                i[:-5]
                for i in os.listdir(os.path.join(self.out_dir, str(idx)))
                if i.endswith(".mol2")
            ][0]

            row = self.task_df[
                (self.task_df["prepped_pdb_file"].str.contains(inp_prot_filename))
                & (self.task_df["ligand_sdf_file"].str.contains(inp_lig_filename))
            ]

            row = row.iloc[0, :]

            # Separate ligands into their own sdf files per output
            tmp_output_dir = os.path.join(out_dir, "tmp_outputs")

            # Only generate outputs if tmp dir does not exist
            if not os.path.exists(tmp_output_dir) or calc_all == True:
                if not os.path.exists(tmp_output_dir):
                    os.makedirs(tmp_output_dir)

                # Finished examples will have ligand_1.lig files
                lig_1_file = [
                    os.path.join(out_dir, i)
                    for i in os.listdir(out_dir)
                    if "ligand_1.lig" in i
                ]

                for lig_file_ in lig_1_file:
                    with open(lig_file_, "rb") as lig_file:
                        readlines = lig_file.readlines()

                        start_mol = False
                        start_atom = False
                        end_mol = False

                        num_atoms = None
                        num_bonds = None

                        mol2_file = []

                        mol_counter = 1
                        line_idx = 0
                        mol_line_idx = None

                        for line in readlines:
                            line = line.decode("ascii")
                            if "<MOLECULE>" in line:
                                line = (
                                    "@<TRIPOS>MOLECULE\n"
                                    + "molecule_"
                                    + str(mol_counter)
                                    + "\n"
                                )
                                start_mol = True

                            if "<ATOM>" in line:
                                num_atoms = line.split("<ATOM>")[-1].strip("\n")
                                line = "@<TRIPOS>ATOM\n"
                                start_atom = True

                            if "<BOND>" in line:
                                num_bonds = line.split("<BOND>")[-1].strip("\n")
                                line = "@<TRIPOS>BOND\n"

                            if start_mol and not start_atom:
                                if "@<TRIPOS>MOLECULE" in line:
                                    mol_line_idx = line_idx
                                    mol2_file.append(line)
                                else:
                                    line = "#" + line
                                    mol2_file.append(line)

                                line_idx += 1

                            if "<INDEX>" in line:
                                end_mol = True

                            if start_atom and not end_mol:
                                line_idx += 1
                                mol2_file.append(line)

                            if "<END>" in line:
                                line_idx += 1

                                if num_atoms is not None and num_bonds is not None:
                                    mol2_file[mol_line_idx] = (
                                        mol2_file[mol_line_idx]
                                        + num_atoms
                                        + num_bonds
                                        + " 0 0 0\n"
                                        + "SMALL\n"
                                        + "GASTEIGER\n"
                                    )

                                start_mol = False
                                start_atom = False
                                end_mol = False

                                num_atoms = None
                                num_bonds = None

                                line_idx = 0
                                mol_line_idx = None

                                if len(mol2_file) != 0:
                                    tmp_filename = (
                                        "".join(
                                            random.choice(string.ascii_lowercase)
                                            for i in range(42)
                                        )
                                        + ".mol2"
                                    )
                                    with open(tmp_filename, "w") as out_mol2_file:
                                        out_mol2_file.write("".join(mol2_file))

                                    mol = Chem.rdmolfiles.MolFromMol2File(tmp_filename)
                                    if mol is not None:
                                        writer = Chem.SDWriter(
                                            os.path.join(
                                                tmp_output_dir,
                                                str(mol_counter) + ".sdf",
                                            )
                                        )
                                        writer.write(mol)

                                    os.remove(tmp_filename)

                                    mol_counter += 1

                                    mol2_file = []

            sdf_files = [
                os.path.join(tmp_output_dir, i)
                for i in os.listdir(tmp_output_dir)
                if i.endswith(".sdf")
            ]

            if len(sdf_files) != 0:
                for sdf_file in sdf_files:
                    PDBs.append(row["PDB ID"])
                    prepped_files.append(row["prepped_pdb_file"])
                    ligand_files.append(row["ligand_sdf_file"])
                    COM_x.append(row["COM_x"])
                    COM_y.append(row["COM_y"])
                    COM_z.append(row["COM_z"])
                    sdf_path.append(str(sdf_file))
            else:
                PDBs.append(row["PDB ID"])
                prepped_files.append(row["prepped_pdb_file"])
                ligand_files.append(row["ligand_sdf_file"])
                COM_x.append(row["COM_x"])
                COM_y.append(row["COM_y"])
                COM_z.append(row["COM_z"])
                sdf_path.append("Errno")

        task_df_with_outputs["PDB ID"] = PDBs
        task_df_with_outputs["mol_cond"] = prepped_files
        task_df_with_outputs["mol_true"] = ligand_files
        task_df_with_outputs["COM_x"] = COM_x
        task_df_with_outputs["COM_y"] = COM_y
        task_df_with_outputs["COM_z"] = COM_z
        task_df_with_outputs["mol_pred"] = sdf_path

        return task_df_with_outputs

    def df_from_model_dir(self, calc_all=True):
        """
        Creates summarised DataFrame from output directory
        """

        # Pocket2Mol
        if self.model_name.lower() == "pocket2mol":
            output_df = self.extract_p2m_outputs()
        # DiffSBDD
        elif self.model_name.lower() == "diffsbdd":
            output_df = self.extract_diffsbdd_outputs(calc_all=calc_all)
        # AutoGrow4
        elif self.model_name.lower() == "autogrow4":
            output_df = self.extract_autogrow4_outputs(calc_all=calc_all)
        # LigBuilderV3
        elif self.model_name.lower() == "ligbuilderv3":
            output_df = self.extract_ligbuilder_outputs(calc_all=calc_all)
        # If none or other model name
        else:
            raise ValueError("Incorrect model name")

        return output_df

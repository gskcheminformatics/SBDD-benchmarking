import os
import pandas as pd
import sys
import random
import string

from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit.Chem.SpacialScore import SPS
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from Bio import PDB

from posebusters.posebusters import PoseBusters

from moses.metrics.utils import (
    _filters,
    compute_fragments,
    compute_scaffolds,
    logP,
    QED,
    SA,
    weight,
)
from moses.metrics import metrics
from fcd_torch import FCD

from plip.structure.preparation import PDBComplex
from plip.exchange.report import BindingSiteReport

from prot_lig_combine import combine


class recreatedPLIP:
    def __init__(self, pdb_file, lig_file):
        """
        Calculates interaction profiles for given PDB and ligand file

        Parameters
        ----------
            pdb_file: str
                Input PDB file without ligand present
            lig_file: str
                Input ligand file
        """

        self.pdb_file = pdb_file
        self.lig_file = lig_file

        self.interaction_dict = self.calculate_interactions()

    def volkamer_get_sites(self, prot_lig_complex):
        """
        Uses PLIP to find interactions between PDB and ligand given a PDB-ligand complex file
        From Volkamer lab tutorial T016: https://projects.volkamerlab.org/teachopencadd/talktorials/T016_protein_ligand_interactions.html

        Parameters
        ----------
            prot_lig_complex: str
                Filepath of protein-ligand complex

        Returns
        -------
            sites: Dictionary of interactions per type
        """

        protlig = PDBComplex()
        protlig.load_pdb(prot_lig_complex)  # load the pdb file
        for ligand in protlig.ligands:
            protlig.characterize_complex(
                ligand
            )  # find ligands and analyze interactions

        sites = {}

        binding_sites = protlig.interaction_sets.items()
        # There should only be one ligand we are assessing - should have Chain ID 'Z'
        # loop over binding sites
        for key, site in sorted(binding_sites):
            if key.split(":")[1] == "Z":
                binding_site = BindingSiteReport(site)

                # tuples of *_features and *_info will be converted to pandas DataFrame
                keys = (
                    "hydrophobic",
                    "hbond",
                    "waterbridge",
                    "saltbridge",
                    "pistacking",
                    "pication",
                    "halogen",
                    "metal",
                )
                # interactions is a dictionary which contains relevant information for each
                # of the possible interactions: hydrophobic, hbond, etc. in the considered
                # binding site. Each interaction contains a list with
                # 1. the features of that interaction, e.g. for hydrophobic:
                # ('RESNR', 'RESTYPE', ..., 'LIGCOO', 'PROTCOO')
                # 2. information for each of these features, e.g. for hydrophobic
                # (residue nb, residue type,..., ligand atom 3D coord., protein atom 3D coord.)
                interactions = {
                    k: [getattr(binding_site, k + "_features")]
                    + getattr(binding_site, k + "_info")
                    for k in keys
                }
                sites[key] = interactions

        return sites

    def calculate_interactions(self):
        """
        Calculates interactions for each type - hydrophobic, hbond, waterbridge, saltbridge, pistacking, pication, halogen, metal

        Parameters
        ----------
            pdb_file: str
                Path to PDB file without a ligand
            lig_file: str
                Path to ligand SDF file

        Returns
        -------
            Dictionary for hydrophobic, hbond, waterbridge, saltbridge, pistacking, pication, halogen, and metal interactions
        """

        interaction_types = {
            "hydrophobic": pd.DataFrame(columns=['RESNR', 'DIST']),
            "hbond": pd.DataFrame(columns=['RESNR', 'DON_ANGLE', 'DIST_H-A', 'DIST_D-A']),
            "waterbridge": pd.DataFrame(columns=['RESNR', 'DIST_A-W', 'DIST_D-W']),
            "saltbridge": pd.DataFrame(columns=['RESNR', 'DIST']),
            "pistacking": pd.DataFrame(columns=['RESNR', 'angle']),
            "pication": pd.DataFrame(columns=['RESNR', 'DIST']),
            "halogen": pd.DataFrame(columns=['RESNR', 'DIST']),
            "metal": pd.DataFrame(columns=['RESNR', 'DIST', 'RMS']),
        }

        try:
            # Combine ligand and protein into one file
            tmp_filename = (
                "".join(random.choice(string.ascii_lowercase) for i in range(42))
                + ".pdb"
            )
            combine.combine(self.pdb_file, self.lig_file, tmp_filename)

            # Use PLIP to calculate interactions
            sites = self.volkamer_get_sites(prot_lig_complex=tmp_filename)

            os.remove(tmp_filename)

            for interaction_type in list(interaction_types.keys()):
                # Extract ligand
                lig_id = list(sites.keys())[0]
                df = pd.DataFrame.from_records(
                    sites[lig_id][interaction_type][1:],
                    columns=sites[lig_id][interaction_type][0],
                )

                interaction_types[interaction_type] = pd.concat(
                    [interaction_types[interaction_type], df]
                )
        except:
            interaction_types = interaction_types

        return interaction_types

    def get_ave(self, inp_list):
        """
        Gets average of list if not empty, otherwise returns None

        Parameters
        ----------
            inp_list: list
                Inputs from PLIP in list format

        Returns
        -------
            ave_list: float average of input list if list has values, otherwise None
        """

        inp_list_float = [float(i) for i in inp_list]

        if len(inp_list_float) != 0:
            ave_list = sum(inp_list_float) / len(inp_list_float)
            return ave_list

        return None

    def recreated_interactions(self, interactions_real, interactions_gen):
        """
        Calculates percentage of interactions re-created for each interaction type

        Parameters
        ---------
            interactions_real: dict
                Dictionary containing interactions of the real protein-ligand complex from the calculate_interactions() function
            interactions_gen: dit
                Dictionary containing interactions of the generated molecule from the calculate_interactions() function

        Returns
        -------
            frac_recreated: dict
                Dictionary of fraction of interactions and distances/angles re-created compared to the ground truth crystal structure for each interaction type
        """

        frac_recreated = {
            "hydrophobic": {
                "interaction_frac": None,
                "ave_dist_real": None,
                "ave_dist_gen": None,
            },
            "hbond": {
                "interaction_frac": None,
                "ave_angle_real": None,
                "ave_angle_gen": None,
                "ave_HA_dist_real": None,
                "ave_HA_dist_gen": None,
                "ave_HD_dist_real": None,
                "ave_HD_dist_gen": None,
            },
            "waterbridge": {
                "interaction_frac": None,
                "ave_dist_a_real": None,
                "ave_dist_a_gen": None,
                "ave_dist_d_real": None,
                "ave_dist_d_gen": None,
            },
            "saltbridge": {
                "interaction_frac": None,
                "ave_dist_real": None,
                "ave_dist_gen": None,
            },
            "pistacking": {
                "interaction_frac": None,
                "ave_angle_real": None,
                "ave_angle_gen": None,
            },
            "pication": {
                "interaction_frac": None,
                "ave_dist_real": None,
                "ave_dist_gen": None,
            },
            "halogen": {
                "interaction_frac": None,
                "ave_dist_real": None,
                "ave_dist_gen": None,
            },
            "metal": {
                "interaction_frac": None,
                "ave_dist_real": None,
                "ave_dist_gen": None,
                "ave_rms_real": None,
                "ave_rms_gen": None,
            },
        }

        for interaction_type in list(frac_recreated.keys()):
            if "RESNR" in interactions_real[interaction_type].columns:
                residues_interacting_real = interactions_real[interaction_type][
                    "RESNR"
                ].tolist()
            else:
                residues_interacting_real = []

            if len(residues_interacting_real) != 0:
                # Average distance of interactions
                if (
                    interaction_type == "hydrophobic"
                    or interaction_type == "saltbridge"
                    or interaction_type == "pication"
                    or interaction_type == "halogen"
                ):
                    if "DIST" in interactions_real[interaction_type].columns:
                        frac_recreated[interaction_type][
                            "ave_dist_real"
                        ] = self.get_ave(
                            interactions_real[interaction_type]["DIST"].tolist()
                        )
                        frac_recreated[interaction_type]["ave_dist_gen"] = self.get_ave(
                            interactions_gen[interaction_type]["DIST"].tolist()
                        )

                elif interaction_type == "hbond":
                    if "DON_ANGLE" in interactions_real[interaction_type].columns:
                        frac_recreated[interaction_type][
                            "ave_angle_real"
                        ] = self.get_ave(
                            interactions_real[interaction_type]["DON_ANGLE"].tolist()
                        )
                        frac_recreated[interaction_type][
                            "ave_angle_gen"
                        ] = self.get_ave(
                            interactions_gen[interaction_type]["DON_ANGLE"].tolist()
                        )
                    if "DIST_H-A" in interactions_real[interaction_type].columns:
                        frac_recreated[interaction_type][
                            "ave_HA_dist_real"
                        ] = self.get_ave(
                            interactions_real[interaction_type]["DIST_H-A"].tolist()
                        )
                        frac_recreated[interaction_type][
                            "ave_HA_dist_gen"
                        ] = self.get_ave(
                            interactions_gen[interaction_type]["DIST_H-A"].tolist()
                        )
                    if "DIST_D-A" in interactions_real[interaction_type].columns:
                        frac_recreated[interaction_type][
                            "ave_HD_dist_real"
                        ] = self.get_ave(
                            interactions_real[interaction_type]["DIST_D-A"].tolist()
                        )
                        frac_recreated[interaction_type][
                            "ave_HD_dist_gen"
                        ] = self.get_ave(
                            interactions_gen[interaction_type]["DIST_D-A"].tolist()
                        )

                elif interaction_type == "waterbridge":
                    if "DIST_A-W" in interactions_real[interaction_type].columns:
                        frac_recreated[interaction_type][
                            "ave_dist_a_real"
                        ] = self.get_ave(
                            interactions_real[interaction_type]["DIST_A-W"].tolist()
                        )
                        frac_recreated[interaction_type][
                            "ave_dist_a_gen"
                        ] = self.get_ave(
                            interactions_gen[interaction_type]["DIST_A-W"].tolist()
                        )
                    if "DIST_D-W" in interactions_real[interaction_type].columns:
                        frac_recreated[interaction_type][
                            "ave_dist_d_real"
                        ] = self.get_ave(
                            interactions_real[interaction_type]["DIST_D-W"].tolist()
                        )
                        frac_recreated[interaction_type][
                            "ave_dist_d_gen"
                        ] = self.get_ave(
                            interactions_gen[interaction_type]["DIST_D-W"].tolist()
                        )

                elif interaction_type == "pistacking":
                    if "angle" in interactions_real[interaction_type].columns:
                        frac_recreated[interaction_type][
                            "ave_angle_real"
                        ] = self.get_ave(
                            interactions_real[interaction_type]["angle"].tolist()
                        )
                        frac_recreated[interaction_type][
                            "ave_angle_gen"
                        ] = self.get_ave(
                            interactions_gen[interaction_type]["angle"].tolist()
                        )

                elif interaction_type == "metal":
                    if "DIST" in interactions_real[interaction_type].columns:
                        frac_recreated[interaction_type][
                            "ave_dist_real"
                        ] = self.get_ave(
                            interactions_real[interaction_type]["DIST"].tolist()
                        )
                        frac_recreated[interaction_type]["ave_dist_gen"] = self.get_ave(
                            interactions_gen[interaction_type]["DIST"].tolist()
                        )
                    if "RMS" in interactions_real[interaction_type].columns:
                        frac_recreated[interaction_type]["ave_rms_real"] = self.get_ave(
                            interactions_real[interaction_type]["RMS"].tolist()
                        )
                        frac_recreated[interaction_type]["ave_rms_gen"] = self.get_ave(
                            interactions_gen[interaction_type]["RMS"].tolist()
                        )

                residues_interacting_gen = interactions_gen[interaction_type][
                    "RESNR"
                ].tolist()

                # Overlap of re-created interactions
                frac_overlap = len(
                    set(residues_interacting_real).intersection(
                        set(residues_interacting_gen)
                    )
                ) / len(set(residues_interacting_real))
            else:
                frac_overlap = None

            frac_recreated[interaction_type]["interaction_frac"] = frac_overlap

        return frac_recreated


class metricCalcs:
    def __init__(
        self,
        output_df,
        training_data_path="Benchmarking_Tasks/Task0/BindingMOAD_filtered_set_for_retraining_with_split.csv",
    ):
        """
        Calculates MOSES, PLIP, and task-specific metrics outlined in the benchmark

        Parameters
        ----------
            output_df: DataFrame
                From getOutputDf class
            training_data_path: str
                Path to retraining csv file containing train/test splits
        """

        # Do not parse through any SDF files that have not been generated
        self.output_df = output_df[
            ~output_df["mol_pred"].astype(str).str.contains("Errno")
        ]
        self.train_df = pd.read_csv(training_data_path)

        # Patterns for Bemis-Murcko scaffold conversion from RDKit
        self.pattern = Chem.MolFromSmarts("[$([D1]=[*])]")
        self.replace = Chem.MolFromSmarts("[*]")

    def sdf_to_smi(self, sdf_file):
        """
        Outputs files from sdf to SMILES

        Parameters
        ----------
            sdf_file: str
                Path to sdf file containing generated compounds

        Returns
        -------
            all_smiles: list of SMILES string/s
        """

        # Get SMILES from SDF files
        all_smiles = []
        smis = [
            Chem.MolToSmiles(mol) for mol in Chem.SDMolSupplier(sdf_file) if mol != None
        ]
        for smi in smis:
            all_smiles.append(smi)

        if len(all_smiles) == 1:
            return all_smiles[0]
        else:
            raise ValueError("More than one molecule in the same SDF file")

    def standardize_cmpds(self, smi_list):
        """
        Standardizes, canonicalizes, and de-salts a list of SMILES

        Parameters
        ----------
            smi_list: list
                SMILES strings to be standardized

        Returns
        -------
            standard_smi: list of standardized SMILES
            standard_mol: list of standardized RDKit Mol objects
        """

        standard_smi = []
        standard_mol = []

        for smi in smi_list:
            mol = Chem.MolFromSmiles(smi)

            try:
                # Standardization: reionize, remove Hydrogens, disconnect metals
                Chem.rdmolops.Cleanup(mol)
                # Checks validity of molecule
                Chem.rdmolops.SanitizeMol(mol)

                # Remove stereochemistry
                Chem.rdmolops.RemoveStereochemistry(mol)

                # Get main molecule if fragments/salts in mol
                mol_parent = Chem.MolStandardize.rdMolStandardize.FragmentParent(mol)

                # Neutralize charges
                mol_neutral = Chem.MolStandardize.rdMolStandardize.Uncharger().uncharge(
                    mol_parent
                )

                standard_smi.append(Chem.MolToSmiles(mol_neutral))
                standard_mol.append(mol_neutral)

            except Exception as e:
                print("Error standardizing:", e)

        return standard_smi, standard_mol

    def get_perc_true(self, inp_list, average=False):
        """
        From a list of booleans, extract number of True

        Parameters
        ----------
            inp_list: list
                Input boolean list
            average: bool
                Optional argument, returns average instead of percentage if set to True

        Returns
        -------
            perc_true: float of number of True against all True and False
        """

        inp_list_no_nan = [i for i in inp_list if str(i) != "nan"]

        try:
            if len(inp_list_no_nan) != 0:
                if average:
                    perc_true = sum(inp_list_no_nan)/len(inp_list_no_nan)
                else:
                    perc_true = (inp_list_no_nan.count(True) / (inp_list_no_nan.count(True) + inp_list_no_nan.count(False)))*100
            else:
                return None
        except:
            return None

        return perc_true

    def df_train_test_smi(self, df, train_test):
        """
        Get all training or testing SMILES given DataFrame from training_data_path

        Parameters
        ----------
            df: pandas DataFrame
                Input training csv with train/test splits
            train_test: str
                Train or test

        Returns
        -------
            smiles: list of training or testing SMILES
        """

        if train_test.lower() == "train":
            smiles = df[df["Train/Test"] == "Train"]["SMILES"].tolist()
        else:
            smiles = df[df["Train/Test"] == "Test"]["SMILES"].tolist()
        return smiles

    def pose_busters_metrics(self, table_to_bust):
        """
        Runs PoseBusters on output SDF files

        Parameters
        ----------
            table_to_bust: pandas DataFrame
                Input DataFrame to use in PoseBusters - from getOutputDf class

        Returns
        -------
            busted_df: pandas DataFrame with full report of "busted" compounds
        """

        pose_busters = PoseBusters()
        busted_df = pose_busters.bust_table(mol_table=table_to_bust, full_report=True)
        return busted_df

    def moses_unique(self, gen_smiles):
        """
        Checks uniqueness @ 1000 given all generated

        Parameters
        ----------
            gen_smiles: list
                Generated SMILES

        Returns
        -------
            unique: float of fraction of unique compounds
        """

        unique = metrics.fraction_unique(gen=gen_smiles, k=1000)
        return unique

    def moses_novelty(self, gen_smiles, train_smiles):
        """
        Checks moses novelty score

        Parameters
        ----------
            gen_smiles: list
                Generated SMILES
            train_smiles: list
                SMILES used for training the model

        Returns
        -------
            novelty: float of fraction of novel compounds
        """

        novelty = metrics.novelty(gen=gen_smiles, train=train_smiles)
        return novelty

    def moses_filters(self, mol):
        """
        Taken from MOSES - checks if molecule passes filters defined in MOSES without the allowed atoms constraint
        Checks for passes of custom medicinal chemistry filters (MCFs) and PAINS filters and does not have charged atoms (avoid ambiguity with tautomers and pH conditions)

        Parameters
        ----------
            mol: RDKit Mol object

        Returns
        -------
            True if passes all checks, otherwise False
        """

        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() != 0 and any(
            len(x) >= 8 for x in ring_info.AtomRings()
        ):
            return False

        h_mol = Chem.AddHs(mol)
        if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
            return False
        if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
            return False

        return True

    def bm_frameworks(self, gen, generic=False):
        """
        Calculates Bemis-Murcko frameworks as detailed https://github.com/rdkit/rdkit/discussions/6844

        Parameters
        ----------
            gen: List
                Generated RDKit mol objects
            generic: bool
                Optional parameter, set to True if generic (graph) frameworks required

        Returns
        -------
            num_frameworks: int
                Number of distinct frameworks in given list of RDKit mol objects
        """

        scaffolds = []

        for mol in gen:
            Chem.RemoveStereochemistry(mol)

            # Get BM scaffold as defined in RDKit
            scaff = MurckoScaffold.GetScaffoldForMol(mol)
            # Replace exo substitutents with electron pair
            scaff = Chem.AllChem.ReplaceSubstructs(
                scaff, self.pattern, self.replace, replaceAll=True
            )[0]

            if generic == True:
                scaff = MurckoScaffold.MakeScaffoldGeneric(scaff)
                scaff = MurckoScaffold.GetScaffoldForMol(scaff)

            # Sanitize and canonicalize scaffold mol objects
            Chem.SanitizeMol(scaff)
            scaff_canon_smi = Chem.MolToSmiles(scaff)

            scaffolds.append(scaff_canon_smi)

        return list(set(scaffolds))

    def per_row_metrics(self, df):
        """
        Calculate metrics per compound

        Parameters
        ----------
            df: pandas DataFrame
                Generated compounds per protein-ligand input

        Returns
        -------
            out_df: pandas DataFrame of input information with SA, validity, filter checks, nSPS, and protein-ligand interaction scores
        """

        PDBs = []
        prepped_files = []
        ligand_files = []
        COM_x = []
        COM_y = []
        COM_z = []
        sdf_path = []

        SA_score = []
        valid = []
        moses_filters = []
        nSPS = []
        ring_info = []
        ring_sizes = []
        interaction_dict = []

        # Should have the same input protein and crystal ligand across all rows
        init_interactions = recreatedPLIP(
            df.iloc[0]["mol_cond"], df.iloc[0]["mol_true"]
        )
        real_interactions = init_interactions.interaction_dict

        for idx, row in df.iterrows():
            mol_sdf = row["mol_pred"]
            # Calculate protein-ligand interaction overlap
            gen_interactions = recreatedPLIP(row["mol_cond"], mol_sdf).interaction_dict
            try:
                recreated_interactions = init_interactions.recreated_interactions(
                    real_interactions, gen_interactions
                )
                interaction_dict.append(recreated_interactions)
            except:
                interaction_dict.append(None)

            with Chem.SDMolSupplier(mol_sdf) as suppl:
                for mol in suppl:
                    PDBs.append(row["PDB ID"])
                    prepped_files.append(row["mol_cond"])
                    ligand_files.append(row["mol_true"])
                    COM_x.append(row["COM_x"])
                    COM_y.append(row["COM_y"])
                    COM_z.append(row["COM_z"])
                    sdf_path.append(mol_sdf)

                    if mol is not None:
                        SA_score.append(sascorer.calculateScore(mol))

                        # Validity check
                        try:
                            Chem.SanitizeMol(mol)
                            valid.append(True)
                        except ValueError:
                            valid.append(False)

                        # MOSES filters check
                        moses_filters.append(self.moses_filters(mol=mol))
                        # nSPS - normalized spacial score
                        nSPS.append(SPS(mol=mol))
                        # Ring info
                        ringobj = mol.GetRingInfo()
                        ring_info.append(ringobj.BondRings())
                        ring_sizes.append(ringobj.NumRings())

                    else:
                        SA_score.append(None)
                        valid.append(None)
                        moses_filters.append(None)
                        nSPS.append(None)
                        ring_info.append(None)
                        ring_sizes.append(None)

        out_df = pd.DataFrame()

        out_df["PDB ID"] = PDBs
        out_df["mol_cond"] = prepped_files
        out_df["mol_true"] = ligand_files
        out_df["COM_x"] = COM_x
        out_df["COM_y"] = COM_y
        out_df["COM_z"] = COM_z
        out_df["mol_pred"] = sdf_path

        out_df["SA_score"] = SA_score
        out_df["valid"] = valid
        out_df["moses_filters"] = moses_filters
        out_df["nSPS"] = nSPS
        out_df["ring_info"] = ring_info
        out_df["ring_size"] = ring_sizes
        out_df["PLIP"] = interaction_dict

        return out_df

    def run_metrics(self):
        """
        Gets all aggregated and per-compound generated molecule metrics per PDB file-ligand file input combination

        Returns
        -------
            aggregated_scores: pandas DataFrame of aggregated MOSES metrics
            per_compound_scores: pandas DataFrame of PoseBusters outputs and PLIP metrics for each generated compound

        """

        per_compound_scores = pd.DataFrame()
        aggregated_scores = pd.DataFrame()

        train_smiles = self.df_train_test_smi(df=self.train_df, train_test="Train")
        test_smiles = self.df_train_test_smi(df=self.train_df, train_test="Test")

        test_mols = [Chem.MolFromSmiles(smi) for smi in test_smiles]
        test_mols = [i for i in test_mols if i is not None]
        # Test fragments
        test_frags = compute_fragments(mol_list=test_mols)
        # Test scaffolds
        test_scaf = compute_scaffolds(mol_list=test_mols)

        # Go through each of the testing examples
        for idx, row in (
            self.output_df[["PDB ID", "mol_cond", "mol_true"]]
            .drop_duplicates()
            .iterrows()
        ):
            # Per row metrics
            print("Running per-row metrics")
            df_per_comp = self.output_df[
                (self.output_df["mol_cond"] == row["mol_cond"])
                & (self.output_df["mol_true"] == row["mol_true"])
            ]
            row_metrics_df = self.per_row_metrics(df=df_per_comp)

            # Get real crystal ligand location
            real_lig_file = row["mol_true"]

            # Get the generated molecules for given pdb and sdf inputs
            gen_files = df_per_comp["mol_pred"].tolist()
            gen_smiles = [self.sdf_to_smi(sdf_file=i) for i in gen_files]
            # Clean-up generated molecules
            gen_smiles, gen_mols = self.standardize_cmpds(gen_smiles)

            print("Running aggregate metrics")
            if True:
                # Per sub-df metrics
                # 1. Uniqueness
                uniqueness = self.moses_unique(gen_smiles=gen_smiles)
                # 2. Novelty
                novelty = self.moses_novelty(
                    gen_smiles=gen_smiles, train_smiles=train_smiles
                )
                # 3. Fragment similarity
                try:
                    sample_frags = compute_fragments(mol_list=gen_mols)
                except Exception as e:
                    [Chem.SanitizeMol(mol) for mol in gen_mols]
                    sample_frags = compute_fragments(mol_list=gen_mols)
                frag_sim = metrics.cos_similarity(test_frags, sample_frags)
                # 4. Scaffold similarity
                sample_scaf = compute_scaffolds(mol_list=gen_mols)
                scaf_sim = metrics.cos_similarity(test_scaf, sample_scaf)
                # 5. FCD
                fcd = FCD()
                fcd_metric = fcd(gen_smiles, test_smiles)
                # 6. KL divergence
                # logP
                wm = metrics.WassersteinMetric(func=logP)
                pref = wm.precalc(test_mols)
                pgen = wm.precalc(gen_mols)
                logp_divergence = wm.metric(pref, pgen)
                # SA
                wm = metrics.WassersteinMetric(func=SA)
                pref = wm.precalc(test_mols)
                pgen = wm.precalc(gen_mols)
                sa_divergence = wm.metric(pref, pgen)
                # QED
                wm = metrics.WassersteinMetric(func=QED)
                pref = wm.precalc(test_mols)
                pgen = wm.precalc(gen_mols)
                qed_divergence = wm.metric(pref, pgen)
                # MW
                wm = metrics.WassersteinMetric(func=weight)
                pref = wm.precalc(test_mols)
                pgen = wm.precalc(gen_mols)
                mw_divergence = wm.metric(pref, pgen)
                # 7. SNN
                snn = metrics.SNNMetric()
                pref = snn.precalc(test_mols)
                pgen = snn.precalc(gen_mols)
                snn_metric = snn.metric(pref, pgen)
                # 8. Internal diversity
                int_div = metrics.internal_diversity(gen=gen_mols)
                # 9. Coverage with frameworks
                num_distinct_atomic_frameworks = len(set(self.bm_frameworks(gen=gen_mols)))
                num_distinct_graph_frameworks = len(
                    set(self.bm_frameworks(gen=gen_mols, generic=True))
                )
                # 10. PoseBusters
                busted_df = self.pose_busters_metrics(
                    table_to_bust=df_per_comp
                ).reset_index()
                busted_df.rename(columns={"file": "mol_pred"}, inplace=True)

                # Combine per row PLIP and posebusters metrics
                per_row_pb_plip = busted_df.merge(
                    row_metrics_df, on="mol_pred", how="inner"
                )
                per_compound_scores = pd.concat([per_compound_scores, per_row_pb_plip])

                # Combine aggregated scores into final DataFrame
                aggregate = pd.DataFrame([row[["PDB ID", "mol_cond", "mol_true"]]])
                aggregate["moses_uniqueness"] = uniqueness
                aggregate["moses_novelty"] = novelty
                aggregate["moses_frag_sim"] = frag_sim
                aggregate["moses_scaf_sim"] = scaf_sim
                aggregate["moses_fcd_metric"] = fcd_metric
                aggregate["moses_logp_div"] = logp_divergence
                aggregate["moses_sa_div"] = sa_divergence
                aggregate["moses_qed_div"] = qed_divergence
                aggregate["moses_mw_div"] = mw_divergence
                aggregate["moses_SNN"] = snn_metric
                aggregate["moses_int_div"] = int_div
                aggregate["moses_coverage_atomic_frameworks"] = num_distinct_atomic_frameworks
                aggregate["moses_coverage_graph_frameworks"] = num_distinct_graph_frameworks

                # Add some per-row metrics
                # PoseBusters
                # 1. Chemical validity and consistency
                aggregate["perc_sanitized"] = self.get_perc_true(per_compound_scores["sanitization"].tolist())
                aggregate["perc_all_atoms_connected"] = self.get_perc_true(per_compound_scores["all_atoms_connected"].tolist())
                aggregate["perc_double_bond_stereo"] = self.get_perc_true(per_compound_scores["double_bond_stereochemistry"].tolist())
                aggregate["perc_chirality"] = self.get_perc_true(per_compound_scores["tetrahedral_chirality"].tolist())

                # 2. Intramolecular validity
                aggregate["perc_pass_bond_lengths"] = self.get_perc_true(per_compound_scores["bond_lengths"].tolist())
                aggregate["perc_pass_bond_angles"] = self.get_perc_true(per_compound_scores["bond_angles"].tolist())
                aggregate["perc_pass_internal_steric_clash"] = self.get_perc_true(per_compound_scores["internal_steric_clash"].tolist())
                aggregate["perc_flat_aromatic_rings"] = self.get_perc_true(per_compound_scores["aromatic_ring_flatness"].tolist())
                aggregate["perc_planar_double_bonds"] = self.get_perc_true(per_compound_scores["double_bond_flatness"].tolist())
                aggregate["perc_passed_internal_energy"] = self.get_perc_true(per_compound_scores["internal_energy"].tolist())

                # 3. Intermolecular validity
                aggregate["perc_pass_dist_to_prot"] = self.get_perc_true(per_compound_scores["minimum_distance_to_protein"].tolist())
                aggregate["perc_pass_dist_to_org_cofactors"] = self.get_perc_true(per_compound_scores["minimum_distance_to_organic_cofactors"].tolist())
                aggregate["perc_pass_dist_to_inorg_cofactors"] = self.get_perc_true(per_compound_scores["minimum_distance_to_inorganic_cofactors"].tolist())
                aggregate["perc_pass_dist_to_water"] = self.get_perc_true(per_compound_scores["minimum_distance_to_waters"].tolist())
                aggregate["perc_pass_vol_overlap_with_prot"] = self.get_perc_true(per_compound_scores["volume_overlap_with_protein"].tolist())
                aggregate["perc_pass_vol_overlap_with_org_cofactors"] = self.get_perc_true(per_compound_scores["volume_overlap_with_organic_cofactors"].tolist())
                aggregate["perc_pass_vol_overlap_with_inorg_cofactors"] = self.get_perc_true(per_compound_scores["volume_overlap_with_inorganic_cofactors"].tolist())
                # MOSES
                # Check validity
                aggregate["perc_pass_validity"] = self.get_perc_true(per_compound_scores["valid"].tolist())
                # MOSES filters
                aggregate["perc_pass_moses_filters"] = self.get_perc_true(per_compound_scores["moses_filters"].tolist())
                # SA score and nSPS average
                aggregate["ave_SA_score"] = self.get_perc_true(per_compound_scores["SA_score"].tolist(), average=True)
                aggregate["ave_nSPS"] = self.get_perc_true(per_compound_scores["nSPS"].tolist(), average=True)
                # Average number of rings
                aggregate["ave_num_rings"] = self.get_perc_true(per_compound_scores["ring_size"].tolist(), average=True)

                aggregated_scores = pd.concat([aggregated_scores, aggregate])
            #except Exception as e:
            #    print(e)
            #    continue

        return aggregated_scores, per_compound_scores

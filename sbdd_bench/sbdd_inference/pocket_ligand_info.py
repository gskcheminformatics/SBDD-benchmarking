import os
import pandas as pd
from openbabel import pybel
from Bio.PDB import PDBParser
from rdkit import Chem
from sbdd_bench.sbdd_analysis.constants import BLIND_SET_POCKET_IDS, BLIND_SET_PDBS_TO_POCKET_IDS
class getPocketLigandInfo:
    """Opens protein and molecule files in given data directory to extract information on pockets and ligand positions"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # Need to be in these naming formats for proteins
        self.prepped_pdb_files = [
            i
            for i in os.listdir(self.data_dir)
            if (i.endswith("_prepped.pdb") or i.endswith("prepared_protein.pdb"))
        ]
        self.pdb_ids = list(
            set([i.split("_")[0].lower() for i in self.prepped_pdb_files])
        )

    def get_ligands_for_protein(self, pdb):
        """Gets ligands associated with protein. If no ligand found, return None. If more than one ligand found, returns all"""

        ligands = [
            i
            for i in os.listdir(self.data_dir)
            if (i.startswith(pdb.lower()) or i.startswith(pdb.upper()))
            and i.endswith(".mol2")
            and "lig" in i
        ]

        if len(ligands) == 0:
            return None
        return ligands

    def get_similar_protein_ligands(self, pdb):
        """If ligand not available for a PDB, finds alternative with same pocket. Uses information from BLIND_SET_POCKET_IDS and BLIND_SET_PDBS_TO_POCKET_IDS"""

        # Use the inverted dictionary to find the pocket ID for PDB then use this ID to get back the PDBs that are associated with that protein
        all_pdbs_for_protein = [
            i.strip().strip("'")
            for i in BLIND_SET_POCKET_IDS[
                BLIND_SET_PDBS_TO_POCKET_IDS[pdb.lower()]
            ].split(",")
        ]
        # Take the first protein which is not the input pdb, keep looping until find one with ligands, otherwise return None
        all_pdbs_for_protein.remove(pdb)
        for new_pdb in all_pdbs_for_protein:
            new_ligands = self.get_ligands_for_protein(pdb=new_pdb)
            if len(new_ligands) > 0:
                return new_ligands

        return None

    def lig_mol2_to_pdb_sdf(self, lig_mol2_path):
        # Open .mol2 ligand file
        ligand = Chem.MolFromMol2File(lig_mol2_path, removeHs=False)

        # If ligand is None, use openbabel converter instead
        if ligand is None:
            inp = pybel.readfile("mol2", lig_mol2_path)
            out_sdf = pybel.Outputfile(
                "sdf", lig_mol2_path[:-4] + "sdf", overwrite=True
            )
            out_pdb = pybel.Outputfile(
                "pdb", lig_mol2_path[:-4] + "pdb", overwrite=True
            )
            for mol in inp:
                out_sdf.write(mol)
                out_pdb.write(mol)
        else:
            # Save as sdf file for DiffSBDD
            writer = Chem.SDWriter(lig_mol2_path[:-4] + "sdf")
            writer.write(ligand)
            # Save as pdb file
            Chem.MolToPDBFile(ligand, lig_mol2_path[:-4] + "pdb")

    def lig_com(self, lig_pdb_path):
        """Geometric center of ligand in pocket"""
        parser = PDBParser()
        input_molecule = parser.get_structure(file=lig_pdb_path, id="pdb_lig")
        return input_molecule.center_of_mass(geometric=True).tolist()

    def get_com_per_ligand(self, lig_mol2_path):
        # Convert to sdf and pdb files
        self.lig_mol2_to_pdb_sdf(lig_mol2_path=lig_mol2_path)

        # Get COM with pdb file
        ligand_com_xyz = self.lig_com(lig_pdb_path=lig_mol2_path[:-4] + "pdb")
        return ligand_com_xyz

    def task_runs_dataframe(self):
        """For all proteins in task, create DataFrame with PDB, ligand, COM"""

        prepared_pdbs_for_df = []
        pdbs_for_df = []
        ligs_for_df = []
        com_x_per_ligand_for_df = []
        com_y_per_ligand_for_df = []
        com_z_per_ligand_for_df = []

        # For each protein in the task, find all ligands associated with it
        for pdb in self.pdb_ids:
            ligands = self.get_ligands_for_protein(pdb=pdb)

            # If no ligand is available for a particular protein, get pocket information by looking at next similar protein. This is only the case for blind set so should be in BLIND_SET_POCKET_IDS
            if ligands is None:
                # Get ligands for similar protein. Skip pdb if no ligands found
                new_ligands = self.get_similar_protein_ligands(pdb=pdb)
                if new_ligands is None:
                    continue

                ligands = new_ligands

            # For each of these ligands, convert to .pdb and .sdf format then calculate COM
            for ligand in ligands:
                ligname = ligand[:-5]

                com_xyz = self.get_com_per_ligand(
                    lig_mol2_path=os.path.join(self.data_dir, ligand)
                )

                for prepared_pdb in self.prepped_pdb_files:
                    if prepared_pdb.split("_")[0].lower() == pdb:
                        com_x_per_ligand_for_df.append(com_xyz[0])
                        com_y_per_ligand_for_df.append(com_xyz[1])
                        com_z_per_ligand_for_df.append(com_xyz[2])
                        ligs_for_df.append(
                            os.path.join(self.data_dir, ligname + ".sdf")
                        )
                        pdbs_for_df.append(pdb)
                        prepared_pdbs_for_df.append(
                            os.path.join(self.data_dir, prepared_pdb)
                        )

        # Store information in a DataFrame
        df = pd.DataFrame()
        df["PDB ID"] = pdbs_for_df
        df["prepped_pdb_file"] = prepared_pdbs_for_df
        df["ligand_sdf_file"] = ligs_for_df
        df["COM_x"] = com_x_per_ligand_for_df
        df["COM_y"] = com_y_per_ligand_for_df
        df["COM_z"] = com_z_per_ligand_for_df

        return df

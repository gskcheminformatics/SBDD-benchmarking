# Changes made to scripts:
- DiffSBDD <br />
&emsp; Only grabs pocket residues https://github.com/arneschneuing/DiffSBDD/blob/30358af24215921a869619e9ddf1e387cafceedd/lightning_modules.py#L726. This was edited to accept other atoms close in proximity to ligand by editing utils.py. Also replaced unknown atoms with C in lightning_modules.py <br />

- Pocket2Mol <br />
&emsp; Ignores Hydrogen atoms in the pocket and does not consider waters and cofactors https://github.com/pengxingang/Pocket2Mol/blob/main/sample_for_pdb.py#L31. Assume that the residue is actually an Alanine, and add all the elements accordingly. Additional metals etc. that are not part of [6, 7, 8, 16, 34] atomic numbers (excluding H) are changed to C for use during protein featurization in sample_for_pdb.py <br />

- AutoGrow4 <br />
&emsp; Have to use -U flag in prepare_receptor4.py to not do protein preparation as it is set by default. mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py edited with -U '' to avoid protein preparation step so metals and waters are not removed </br>

- LigBuilderv3 <br />
&emsp; Water molecules and metals can be included from the prepared file by using parameters in cavitydefault.input (HETMETAL and HETWATER set to 1) <br />

import os
import tempfile

import deepchem as dc
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem



def prepare_protien(
    protein: str,
    replace_nonstandard_residues: bool = True,
    remove_heterogens: bool = True,
    remove_water: bool = True,
    add_hydrogens: bool = True,
    pH: float = 7.0,
):
    if protein.endswith(".pdb"):
        fixer = PDBFixer(protein)
    else:
        fixer = PDBFixer(url="https://files.rcsb.org/download/%s.pdb" % (protein))

    # Apply common fixes to PDB files
    if replace_nonstandard_residues:
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
    if remove_heterogens and not remove_water:
        fixer.removeHeterogens(True)
    if remove_heterogens and remove_water:
        fixer.removeHeterogens(False)
    if add_hydrogens:
        fixer.addMissingHydrogens(pH)

    tmp_file = tempfile.NamedTemporaryFile('w+', delete=True)
    PDBFile.writeFile(fixer.topology, fixer.positions, tmp_file)
    p = Chem.MolFromPDBFile(tmp_file.name, sanitize=True)
    tmp_file.close()
    return p


def prepare_ligand(ligand: str, optimize_ligand: bool = True):
    if ligand.endswith(".pdb"):
        m = Chem.MolFromPDBFile(ligand)
    else:
        m = Chem.MolFromSmiles(ligand, sanitize=True)

    # Optimize ligand
    if optimize_ligand:
        m = Chem.AddHs(m)  # need hydrogens for optimization
        AllChem.EmbedMolecule(m)
        AllChem.MMFFOptimizeMolecule(m)

    return m


def get_docking_score(protien_mol, ligand_mol):
    try:
        base_dir = tempfile.TemporaryDirectory(prefix="dock")
        base_dir_path = base_dir.name

        tmp_ligand_path = os.path.join(base_dir_path, "ligand.pdb")
        tmp_protien_path = os.path.join(base_dir_path, "protien.pdb")

        Chem.rdmolfiles.MolToPDBFile(ligand_mol, tmp_ligand_path)
        Chem.rdmolfiles.MolToPDBFile(protien_mol, tmp_protien_path)

        vina_dump_path = os.path.join(base_dir_path, "dump")
        os.mkdir(vina_dump_path)

        vpg = dc.dock.pose_generation.VinaPoseGenerator()
        _, scores = vpg.generate_poses(
            molecular_complex=(
                tmp_protien_path,
                tmp_ligand_path,
            ),  # protein-ligand files for docking,
            out_dir=vina_dump_path,
            generate_scores=True,
            exhaustiveness=4,
            verbosity=0,
            seed=42
        )

        base_dir.cleanup()
        return scores[0]
    except Exception as e:
        return None


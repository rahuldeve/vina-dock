
import pandas as pd
from tqdm.auto import tqdm
from utils import prepare_ligand, prepare_protien, get_docking_score
import firebase_admin
from firebase_admin import firestore
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from rdkit.rdBase import BlockLogs


def dock_ligands(ligand_chunk, protien_pdb_path):
    with BlockLogs():
        protien_mol = prepare_protien(protien_pdb_path)

        scores = []
        for ligand_smiles in ligand_chunk:
            try:
                ligand_mol = prepare_ligand(ligand_smiles)
                scores.append(get_docking_score(protien_mol, ligand_mol))
            except Exception as e:
                scores.append(None)

        return scores


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]



if __name__ == "__main__":
    app = firebase_admin.initialize_app()
    fdb = firestore.client()


    ligands_path = Path("./KasA_smm_val.csv")
    protien_pdb_path = Path("./KasA_3285_no_min.pdb")

    df = pd.read_csv(ligands_path)
    smiles = df["smiles"]

    collection = fdb.collection(f"{ligands_path.name}-{protien_pdb_path.name}")
    processed_idxs = [d.get().to_dict()['idx'] for d in collection.list_documents()]
    unprocessed_idxs = ~smiles.index.isin(processed_idxs)
    unprocessed_smiles = smiles[unprocessed_idxs]

    unprocessed_smiles_chunked = [chunk for chunk in chunks(unprocessed_smiles, 4)]


    def parallel_inner(ligand_chunk, protien_pdb_path):
        protien_pdb_path = str(protien_pdb_path.absolute())
        scores = dock_ligands(ligand_chunk, protien_pdb_path)
        return list(zip(ligand_chunk.index, scores))

    pool = Pool(8)

    iterable = pool.imap(
        partial(parallel_inner, protien_pdb_path=protien_pdb_path), unprocessed_smiles_chunked
    )

    for result_chunk in tqdm(iterable, total=len(unprocessed_smiles_chunked)):
        batch = fdb.batch()

        for idx, score in result_chunk:
            
            doc_ref = fdb.collection(
                f"{ligands_path.name}-{protien_pdb_path.name}"
            ).document(str(idx))

            batch.set(
                doc_ref,
                {"idx": idx, "score": score},
            )

        
        batch.commit()

    pool.close()






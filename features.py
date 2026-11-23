import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdMolDescriptors, Descriptors, rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

def drug_features(smi):
    '''1024-bit Morgan fingerprint + 7 descriptors -> 1031 features'''
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fp = _gen.GetFingerprintAsNumPy(m).astype(np.float32)
    desc = np.array([
        Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
        rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
        rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m)
    ], dtype=np.float32)
    return np.concatenate([fp, desc])

def protein_features(seq):
    '''amino-acid composition (20) + dipeptide composition (400) -> 420 features'''
    seq = [c for c in seq if c in AA_IDX]
    n = max(len(seq), 1)
    aac = np.zeros(20, dtype=np.float32)
    dpc = np.zeros((20, 20), dtype=np.float32)
    for c in seq:
        aac[AA_IDX[c]] += 1
    for a, b in zip(seq[:-1], seq[1:]):
        dpc[AA_IDX[a], AA_IDX[b]] += 1
    return np.concatenate([aac / n, dpc.flatten() / max(n - 1, 1)])

def pair_features(smi, seq):
    d = drug_features(smi)
    if d is None:
        return None
    return np.concatenate([d, protein_features(seq)]).reshape(1, -1)

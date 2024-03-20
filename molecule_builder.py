from rdkit import Chem
import dgl
from rdkit.Geometry import Point3D
atom_type_to_idx = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
atom_idx_to_type = { v:k for k,v in atom_type_to_idx.items() }


bond_idx_to_type = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
bond_idx_to_type = { i:val for i,val in enumerate(bond_idx_to_type) }

def graph_to_rdmol(g: dgl.DGLGraph):
    positions = g.ndata['pos']
    atom_types = g.ndata['attr'][:, :len(atom_idx_to_type)].argmax(dim=1).tolist()
    atom_types = [atom_idx_to_type[i] for i in atom_types]

    bond_src_idxs, bond_dst_idxs = g.edges()

    # filter edges by those where the source is less than the destination
    mask = bond_src_idxs < bond_dst_idxs
    bond_src_idxs = bond_src_idxs[mask]
    bond_dst_idxs = bond_dst_idxs[mask]

    bond_type_idxs = g.edata['edge_attr'][mask].argmax(dim=1).tolist()

    mol = build_molecule(positions, atom_types, bond_src_idxs, bond_dst_idxs, bond_type_idxs)
    return mol


def build_molecule(positions, atom_types, bond_src_idxs, bond_dst_idxs, bond_type_idxs):
    """Builds a rdkit molecule from the given atom and bond information."""
    # create a rdkit molecule and add atoms to it
    mol = Chem.RWMol()
    for atom_type in atom_types:
        a = Chem.Atom(atom_type)
        mol.AddAtom(a)

    # add bonds to rdkit molecule
    for bond_type_idx, src_idx, dst_idx in zip(bond_type_idxs, bond_src_idxs, bond_dst_idxs):
        src_idx = int(src_idx)
        dst_idx = int(dst_idx)
        bond_type_idx = int(bond_type_idx)
        bond_type = bond_idx_to_type[bond_type_idx]
        mol.AddBond(src_idx, dst_idx, bond_type)

    try:
        mol = mol.GetMol()
    except Chem.KekulizeException:
        return None

    # Set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = positions[i]
        x, y, z = float(x), float(y), float(z)
        conf.SetAtomPosition(i, Point3D(x,y,z))
    mol.AddConformer(conf)

    return mol
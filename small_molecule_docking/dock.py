"""This module contains functions docking a molecule to a receptor using Openeye.

The code is adapted from this repository: https://github.com/inspiremd/Model-generation
"""
from dataclasses import dataclass
from functools import cache
import MDAnalysis as mda
from natsort import natsorted
import numpy as np
from openeye import oechem, oedocking, oeomega
import os
import pandas as pd
import polars as pl
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import sys
import tempfile
import tomllib
from tqdm import tqdm
from typing import Type, TypeVar, Union
import warnings

'''
Functions
'''
PathLike = Union[Path, str]
_T = TypeVar('_T')

def init_mpi():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

class DockReceptor:
    """
    Dock one ligand to one receptor. Stores scores, smiles and poses if asked for.
    """
    def __init__(self,
                 smiles: list[str],
                 receptor_file: PathLike,
                 pdb_dir: PathLike,
                 out: PathLike,
                 max_confs: int,
                 receptor: object | None = None,
                 cutoff: float=0.,
                 temp_dir: PathLike='temp'):
        self.smiles = smiles
        self.receptor_file = receptor_file 
        pdb = pdb_dir / receptor_file.stem.split('_')[0]
        self.pdb = pdb.with_suffix('.pdb')
        self.out = out
        self.out.mkdir(exist_ok=True, parents=True)
        self.max_confs = max_confs
        self.score_cutoff = cutoff
        self.temp_dir = out / temp_dir
        self.temp_dir.mkdir(exist_ok=True)

        if receptor is None: # allow us to pass these in if need be
            self.receptor = self.initialize_receptor(self.read_receptor(receptor_file))
        else:
            self.receptor = receptor

        self.score_dict = {'smiles': [],
                           'scores': [],
                           'dirs': []}
    
    def screen_all(self):
        for lig_id, smile in enumerate(self.smiles):
            conformers = self.gen_conformers(smile)

            score, d = self.dock_compound(smile, conformers, lig_id)
            
            self.score_dict['smiles'].append(smile)
            self.score_dict['scores'].append(score)
            self.score_dict['dirs'].append(d)

    def gen_conformers(self,
                       smiles: str):
        try:
            conformers = self.select_enantiomer(self.from_string(smiles))
        except:
            with tempfile.NamedTemporaryFile(suffix='.pdb', dir=self.temp_dir) as fd:
                self.smi_to_structure(smiles, Path(fd.name))
                conformers = self.from_structure(Path(fd.name))

        return conformers

    def dock_compound(self,
                      smiles: str,
                      conformers: oechem.OEMol,
                      lig_id: int) -> None:
        """Run OpenEye docking on a single ligand, receptor pair.

        Parameters
        ----------
        smiles : str
            A single SMILES string.
        """
        # Dock the ligand conformers to the receptor
        dock, lig = self.dock_conf(self.receptor, conformers, max_poses=self.max_confs)
    
        # Get the docking scores
        best_score = self.ligand_scores(dock, lig)[0]

        out_return = 'NA'
    
        if self.out.exists():
            out = self.out / f'cmpd{lig_id}'
            out.mkdir(exist_ok=True)
            if best_score <= self.score_cutoff:
                sys.stdout.flush()
                
                self.write_ligand(lig, out)
                u = self.create_universe(str(self.pdb))
                pdbs = out / 'pdbs'
                pdbs.mkdir(exist_ok=True)

                try:
                    self.create_trajectory(u, 
                                           out, 
                                           str(pdbs / f'{self.receptor_file.stem}.{lig_id}'))
                except:
                    try:
                        self.create_trajectory(u, 
                                               out, 
                                               str(pdbs / f'{self.receptor_file.stem}.{lig_id}'))
                    except: #(OSError, IOError, ValueError, EOFError):
                        pass

                out_return = str(out)

        return best_score, out_return

    def from_string(self,
                    smiles: str, 
                    isomer: bool=True, 
                    num_enantiomers: int=1) -> oechem.OEMol:
        """
        Generates an set of conformers from a SMILES string
        """
        mol = oechem.OEMol()
        if not oechem.OESmilesToMol(mol, smiles):
            raise ValueError(f"SMILES invalid for string {smiles}")
        else:
            return self.from_mol(mol, isomer, num_enantiomers)
    
    def create_complex(self,
                       protein_universe: mda.Universe, 
                       ligand_pdb: Path) -> mda.Universe:
        self.add_hydrogens(ligand_pdb)

        u2 = mda.Universe(ligand_pdb)
        return mda.Merge(protein_universe.atoms, u2.atoms)

    def create_trajectory(self,
                          protein_universe: mda.Universe, 
                          ligand_dir: Path, 
                          output_basename: str) -> None:
        ligand_files = natsorted(ligand_dir.glob('*.pdb'))
        atom_groups = [self.create_complex(protein_universe, lig)
                       for lig in ligand_files]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            for i, ag in enumerate(atom_groups):
                out_name = Path(output_basename).with_suffix(f'.pose{i}.pdb')
                sel = ag.select_atoms('all')
                with mda.Writer(out_name) as w:
                    w.write(sel)

    @property
    def scores(self):
        return self.score_dict

    @property
    def pose_map(self):
        return self.mapping
    
    @staticmethod
    def smi_to_structure(smiles: str, 
                         output_file: Path, 
                         forcefield: str = "mmff") -> None:
        """Convert a SMILES file to a structure file.
    
        Parameters
        ----------
        smiles : str
            Input SMILES string.
        output_file : Path
            EIther an output PDB file or output SDF file.
        forcefield : str, optional
            Forcefield to use for 3D conformation generation
            (either "mmff" or "etkdg"), by default "mmff".
        """
        # Convert SMILES to RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
    
        # Add hydrogens to the molecule
        mol = Chem.AddHs(mol)
    
        # Generate a 3D conformation for the molecule
        if forcefield == "mmff":
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
        elif forcefield == "etkdg":
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        else:
            raise ValueError(f"Unknown forcefield: {forcefield}")
    
        # Write the molecule to a file
        if output_file.suffix == ".pdb":
            writer = Chem.PDBWriter(str(output_file))
        elif output_file.suffix == ".sdf":
            writer = Chem.SDWriter(str(output_file))
        else:
            raise ValueError(f"Invalid output file extension: {output_file}")

        writer.write(mol)
        writer.close()
    
    @staticmethod
    def from_mol(mol: oechem.OEMol, 
                 isomer: bool=True, 
                 num_enantiomers: int=1) -> oechem.OEMol:
        """Generates a set of conformers as an OEMol object
        Inputs:
            mol is an OEMol
            isomers is a boolean controlling whether or not the various diasteriomers of a molecule are created
            num_enantiomers is the allowable number of enantiomers. For all, set to -1
        """
        # Turn off the GPU for omega
        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.GetTorDriveOptions().SetUseGPU(False)
        omega = oeomega.OEOmega(omegaOpts)
    
        out_conf = []
        if not isomer:
            ret_code = omega.Build(mol)
            if ret_code == oeomega.OEOmegaReturnCode_Success:
                out_conf.append(mol)
            else:
                oechem.OEThrow.Warning(
                    "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
                )
    
        elif isomer:
            for enantiomer in oeomega.OEFlipper(mol.GetActive(), 12, True):
                enantiomer = oechem.OEMol(enantiomer)
                ret_code = omega.Build(enantiomer)
                if ret_code == oeomega.OEOmegaReturnCode_Success:
                    out_conf.append(enantiomer)
                    num_enantiomers -= 1
                    if num_enantiomers == 0:
                        break
                else:
                    oechem.OEThrow.Warning(
                        "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
                    )
        return out_conf
    
    @staticmethod
    def from_structure(structure_file: Path) -> oechem.OEMol:
        """
        Generates an set of conformers from a SMILES string
        """
        mol = oechem.OEMol()
        ifs = oechem.oemolistream()
        if not ifs.open(str(structure_file)):
            raise ValueError(f"Could not open structure file: {structure_file}")
    
        if structure_file.suffix == ".pdb":
            oechem.OEReadPDBFile(ifs, mol)
        elif structure_file.suffix == ".sdf":
            oechem.OEReadMDLFile(ifs, mol)
        else:
            raise ValueError(f"Invalid structure file extension: {structure_file}")
    
        return mol
    
    @staticmethod
    def select_enantiomer(mol_list: list[oechem.OEMol]) -> oechem.OEMol:
        return np.random.choice(mol_list, 1)[0]

    @staticmethod
    def initialize_receptor(receptor: oechem.OEMol) -> oechem.OEMol:
        dock = oedocking.OEDock()
        dock.Initialize(receptor)
        return dock
    
    @staticmethod
    def dock_conf(dock: oechem.OEMol, 
                  mol: oechem.OEMol, 
                  max_poses: int = 1) -> tuple[oechem.Mol, oechem.OEMol]:
        lig = oechem.OEMol()
        _ = dock.DockMultiConformerMolecule(lig, mol, max_poses)
        return dock, lig
    
    @staticmethod
    # Returns an array of length max_poses from above. This is the range of scores
    def ligand_scores(dock: oechem.OEMol, 
                      lig: oechem.OEMol) -> list[float]:
        return [dock.ScoreLigand(conf) for conf in lig.GetConfs()]
    
    @staticmethod
    def write_ligand(ligand: oechem.OEMol, 
                     output_dir: Path) -> None:
        # TODO: If MAX_POSES != 1, we should select the top pose to save
        ofs = oechem.oemolostream()
        for it, conf in enumerate(list(ligand.GetConfs())):
            pdb = output_dir / f'{it}.pdb'
            if ofs.open(str(pdb)):
                oechem.OEWriteMolecule(ofs, conf)
                ofs.close()
    
    @staticmethod
    @cache  # Only read the receptor once
    def read_receptor(receptor_oedu_file: Path) -> oechem.OEMol:
        """Read the .oedu file into a GraphMol object."""
        receptor = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(str(receptor_oedu_file), receptor)
        return receptor
    
    @staticmethod
    @cache
    def create_universe(protein_pdb: Path) -> mda.Universe:
        return mda.Universe(protein_pdb)
   
    @staticmethod
    def add_hydrogens(ligand_pdb: Path) -> Path:
        mol = Chem.MolFromPDBFile(ligand_pdb)
        molH = Chem.AddHs(mol, addCoords=True)

        Chem.MolToPDBFile(molH, ligand_pdb)


@dataclass
class RunDocking:
    receptor_dir: PathLike
    score_dir: PathLike
    pdb_dir: PathLike
    pose_dir: PathLike
    temp_storage: PathLike
    smiles: PathLike 
    score_prefix: str 
    max_confs: int
    pose_gen: bool
    batch_size: int 
    score_cutoff: float 
    mpi: bool

    def __post_init__(self):
        self.pose_dir.mkdir(exist_ok=True)
        self.score_dir.mkdir(exist_ok=True)
        self.receptor_files = natsorted([r for r in self.receptor_dir.glob('*.oedu')])
        self.smiles = self.read_in(Path(self.smiles))
        
        receptors = [self.read_receptor(rec) for rec in self.receptor_files]
        self.dock_objs = [self.initialize_receptor(receptor) for receptor in receptors]

        self.pose_map = {} # only need this if we are saving poses but harmless otherwise

        if self.pose_gen == True:
            for receptor_file, receptor in zip(self.receptor_files, receptors):
                receptor_out = self.pose_dir / receptor_file.with_suffix('.pdb').name
                if not receptor_out.exists():
                    self.write_receptor(receptor, receptor_out)
        
        self.rank = None # if we are not using mpi so as not to crash when saving scores csv
        if self.mpi:
            ### Initialize mpi
            self.comm, self.rank, self.size = init_mpi()
            self.smiles = np.array_split(self.smiles, self.size)[self.rank]
            self.score_prefix = f'{self.score_prefix}_rank{self.rank}'

    def dock(self) -> None:
        """
        Dock all receptors to all smiles in batches.
        """
        smiles_batch = np.array_split(self.smiles, len(self.smiles) // self.batch_size)

        self.score = pd.DataFrame()
        for dock_obj, receptor_file in zip(self.dock_objs, self.receptor_files):
            name = receptor_file.stem

            batch_score = pd.DataFrame()
            for batch_id, smiles in tqdm(enumerate(smiles_batch)):
                if not isinstance(smiles, np.ndarray):
                    smiles = np.ndarray(smiles)
                
                smi_series = pd.Series(smiles, name='smiles')
                
                out = self.pose_dir / name / f'batch{batch_id}'
                serial = DockReceptor(smiles,
                                      receptor_file,
                                      self.pdb_dir,
                                      out if self.pose_gen else None,
                                      self.max_confs,
                                      dock_obj,
                                      self.score_cutoff)
                
                serial.screen_all()

                score_series = pd.Series(serial.scores['scores'], name=name)
                
                score_df = pd.concat([smi_series, score_series], axis=1)
                if batch_score.empty:
                    batch_score = score_df
                else:
                    batch_score = pd.concat([batch_score, score_df])
                
            if self.score.empty:
                self.score = batch_score
            else:
                self.score = pd.concat([self.score, batch_score[name]], axis=1)

        self.save_dataframes()

    def save_dataframes(self) -> None:
        csv_file = self.score_dir / f'{self.score_prefix}.csv'
        self.score.to_csv(csv_file, index=False)

    def read_in(self,
                smiles: PathLike) -> list[str]:
        if smiles.suffix == '.csv':
            return pd.read_csv(str(smiles))['smiles'].tolist()
        elif smiles.suffix == '.txt':
            return [smile.strip() for smile in open(smiles).readlines()]
        else:
            raise ValueError(f'File type {smiles.suffix} not yet supported for smiles!')
    
    @staticmethod
    @cache  # Only read the receptor once
    def read_receptor(receptor_oedu_file: Path) -> oechem.OEMol:
        """Read the .oedu file into a GraphMol object."""
        receptor = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(str(receptor_oedu_file), receptor)
        return receptor

    @staticmethod
    def initialize_receptor(receptor: oechem.OEMol) -> oechem.OEMol:
        dock = oedocking.OEDock()
        dock.Initialize(receptor)
        return dock
    
    @staticmethod
    def write_receptor(receptor: oechem.OEMol, 
                       output_path: Path) -> None:
        ofs = oechem.oemolostream()
        if ofs.open(str(output_path)):
            mol = oechem.OEMol()
            contents = receptor.GetComponents(mol)
            oechem.OEWriteMolecule(ofs, mol)
            ofs.close()

    @classmethod
    def from_toml(cls: Type[_T],
                  config_file: PathLike,
                  mpi: bool=False) -> _T:
        config = tomllib.load(open(config_file, 'rb'))
        dirs = config['directories']
        settings = config['settings']
        
        receptor_dir = Path(dirs['receptors'])
        score_dir = Path(dirs['scores'])
        pdb_dir = Path(dirs['pdbs']) 
        pose_dir = Path(dirs['poses'])
        temp_storage = Path(dirs['temp'])
        
        smiles = settings['smiles']
        score_prefix = settings['score_pattern']
        max_confs = settings['max_confs'] 
        pose_gen = settings['pose_gen'] 
        batch_size = settings['batch_size']
        score_cutoff = settings['score_cutoff'] 

        return cls(receptor_dir, score_dir, pdb_dir, pose_dir, temp_storage, 
                   smiles, score_prefix, max_confs, pose_gen, batch_size, 
                   score_cutoff, mpi)
        
if __name__ == "__main__":
    '''
    Running Code
    '''
    import argparse
    
    parser = argparse.ArgumentParser(description='load config file')
    parser.add_argument('-c', '--config', type=Path)
    parser.add_argument('-m', '--mpi', type=bool, default=False)
    args = parser.parse_args()
    
    docker = RunDocking.from_toml(args.config, args.mpi)
    docker.dock()

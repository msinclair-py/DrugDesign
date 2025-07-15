from Bio import PDB 
from chroma import Chroma, Protein, conditioners
import MDAnalysis as mda
import numpy as np
import os
from pathlib import Path
import pip._vendor.tomli as tomllib
import shutil
import torch
from typing import Union, Type, TypeVar

_T = TypeVar('_T')
Hotspot = dict[int, Union[str, list[int]]]
OptPath = Union[Path, str, None]
PathLike = Union[Path, str]
Conditioner = conditioners.Conditioner

class ChromaDesigner:
    def __init__(self,
                 input_dir: PathLike,
                 output_dir: PathLike,
                 hotspot_dir: OptPath,
                 hotspots: list[Hotspot],
                 binder_length: list[int],
                 n_rounds: int,
                 n_backbones: int,
                 n_designs: int,
                 diff_steps: int,
                 model_weights: PathLike,
                 backbone_weights: PathLike,
                 conditioner_weights: PathLike,
                 device: str,
                 store_in_memory: bool=False,
                 memory_migration_freq: int=5):
        self.input_dir = Path(input_dir) if isinstance(input_dir, str) else input_dir
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        self.hotspot_dir = Path(hotspot_dir) if isinstance(hotspot_dir, str) else hotspot_dir
        
        self.pdbs = list(self.input_dir.glob('*.pdb'))

        self.hotspots = hotspots
        self.binder_len = [x for x in range(*binder_length)]
        self.n_rounds = n_rounds
        self.n_bbs = n_backbones
        self.N = n_designs
        self.steps = diff_steps

        self.weights = model_weights
        self.bb_weights = backbone_weights
        self.conditioner_weights = conditioner_weights
        self.device = device
        self.store_in_memory = store_in_memory
        self.mem_freq = memory_migration_freq

        self.model = self.load_model()

    def run_rounds(self):
        """
        Main logic for running rounds of Chroma sampling.
        """
        # make output dir after __init__ incase we are appending
        # parallel runs on the fly
        self.output_dir.mkdir(exist_ok=True, parents=True)

        i = 0
        while i < self.n_rounds:
            try:
                pdb, len_binder, hotspot, scaling = self.choose_inputs()
                vector, magnitude = self.prepare_inflate_conditioner(pdb, 
                                                                     hotspot,
                                                                     scaling)

                protein, mask_aa = self.prepare_protein(pdb, len_binder)
                conditioner_list = self.generate_conditioners(protein, vector, magnitude)
                conditioner = conditioners.ComposedConditioner(conditioner_list)
                print(protein)
                proteins, trajectories = self.design(protein, conditioner, mask_aa)
                self.save(proteins, f'round_{i}', pdb.stem)

                i += 1

                if self.store_in_memory and i % self.mem_freq == 0:
                    self.dump_from_memory()

            except RuntimeError as e: # avoid Intel oneMKL Errors and such
                print(e)
                continue

    def design(self,
               protein: Protein,
               conditioner: Conditioner,
               mask_aa: torch.Tensor) -> list[Protein]:
        """
        Runs a single round of Chroma sampling.

        Arguments:
            protein (Protein): Chroma protein object encoded with X, C, S tensors
            conditioner (Conditioner): Chroma composed conditioner object
            mask_aa (torch.Tensor): A mask tensor to ensure only the binder is sampled

        Returns:
            list[Protein]: A list of Chroma protein objects with diffused binders
        """
        proteins, trajectories = self.model.sample(
            protein_init=protein,
            conditioner=conditioner,
            design_selection=mask_aa,
            langevin_factor=2,
            langevin_isothermal=True,
            inverse_temperature=8.0,
            sde_func='langevin',
            full_output=True,
            steps=self.steps,
            samples=self.n_bbs,
            num_designs=self.N
        )

        return proteins, trajectories

    def save(self,
             proteins: list[Protein],
             _round: str,
             name: str) -> None:
        """
        Saves the output from a round of Chroma sampling as cif files.
        """
        path = self.output_dir / _round
        path.mkdir(exist_ok=True, parents=True)
        
        if self.store_in_memory:
            if not hasattr(self, 'mem_path'):
                self.mem_path = Path('/dev/shm') / self.output_dir.name

            path = self.mem_path / _round
            path.mkdir(exist_ok=True, parents=True)

        if not isinstance(proteins, list):
            proteins = [proteins]

        if isinstance(proteins[0], list):
            proteins = proteins[0]

        for i, protein in enumerate(proteins):
            backbone = f'backbone{i // self.N}'
            design = f'design{i % self.N}'
            output = path / f'{name}_{backbone}_{design}.cif'
            protein.to(str(output))
            self.convert_cif_to_pdb(output)

            output.unlink(missing_ok=True)

    def dump_from_memory(self):
        outbound = self.mem_path.glob('round_*')
        inbound = self.output_dir
        
        for directory in outbound:
            files = directory.glob('*')
            for file in files:
                if not (in_file := (inbound / directory.name / file.name).resolve()).exists():
                    shutil.copy2(str(file), str(in_file))
                    os.remove(str(file))

    def load_model(self) -> Chroma:
        """
        Load Chroma model with weights.
        """
        return Chroma(
            weights_backbone = str(self.bb_weights),
            weights_design = str(self.weights),
            device = self.device
        )

    def choose_inputs(self) -> tuple[PathLike, int, str, float]:
        pdb = self.choose_pdb()
        length = self.choose_binder_length()

        if self.hotspot_dir is not None:
            hotspot_residues = self.choose_hotspot_from_dir(pdb.stem)
            scaling = 1.1 # default, maybe ingest this into the class attributes
        else:
            hotspot_residues, scaling = self.choose_hotspot()

        return pdb, length, hotspot_residues, scaling

    def choose_pdb(self) -> PathLike:
        return np.random.choice(self.pdbs, 1)[0]

    def choose_binder_length(self) -> int:
        return np.random.choice(self.binder_len, 1)[0]

    def choose_hotspot_from_dir(self,
                                name: str) -> str:
        hotspots = self.hotspot_dir.glob(f'{name}_hspot*.txt')
        hotspot_file = np.random_choice(hotspots, 1)[0]
        return open(hotspot_file).readline().strip()

    def choose_hotspot(self) -> tuple[str, float]:
        keys = list(self.hotspots.keys())
        key_index = np.random.choice(len(keys))
        hotspot = self.hotspots[keys[key_index]]

        if hotspot['type'] == 'range':
            hotspot_residues = [x for x in range(*hotspot['indices'])]
        else:
            hotspot_residues = [x for x in hotspot['indices']]

        scaling = hotspot['vector_scaling']

        return (' '.join([str(x) for x in hotspot_residues]), scaling)

    def prepare_inflate_conditioner(self,
                                    pdb: PathLike,
                                    hotspot_residues: str,
                                    vector_scaling: float=1.) -> tuple[torch.Tensor, float]:
        """
        Using MDAnalysis and a definition for hotspot returns a tensor corresponding
        to the hotspot vector and the scaled magnitude.
        """
        u = mda.Universe(str(pdb))
        hspot = u.select_atoms(f'protein and resid {hotspot_residues}')

        vector = hspot.center_of_mass() - u.atoms.center_of_mass()
        magnitude = np.linalg.norm(vector) * vector_scaling

        vector = torch.tensor(vector, dtype=torch.float)

        return vector, magnitude

    def prepare_protein(self,
                        pdb: PathLike,
                        len_binder: int) -> tuple[Protein, torch.Tensor]:
        """
        Adds to the input protein tensors to accomodate a binder of length 
        `len_binder`. Returns protein object and masking tensor.
        """
        protein = Protein(str(pdb), device=self.device)
        X, C, S = protein.to_XCS()

        X_new = torch.cat(
            [X,
             torch.zeros(1, len_binder, 4, 3).xpu()
            ],
            dim=1
        )

        C_new = torch.cat(
            [C,
             torch.full((1, len_binder), 2).xpu()
            ],
            dim=1
        )

        S_new = torch.cat(
            [S,
             torch.full((1, len_binder), 0).xpu()
            ],
            dim=1
        )

        del X, C, S

        protein = Protein(X_new, C_new, S_new, device=self.device)
        X, C, S = protein.to_XCS()
        
        L_binder = (C == 2).sum().item()
        L_receptor = (C == 1).sum().item()
        L_complex = L_binder + L_receptor
        assert L_complex == C.shape[-1] # something terrible went wrong

        mask_aa = torch.Tensor(L_complex * [[0] * 20])
        for i in range(L_complex):
            if i not in range(L_receptor):
                mask_aa[i] = torch.Tensor([1] * 20)
                mask_aa[i][S[0][i].item()] = 1

        mask_aa = mask_aa[None].xpu()
        
        residues_to_keep = [i for i in range(L_receptor)]
        protein.sys.save_selection(gti=residues_to_keep, selname='receptor')

        return protein, mask_aa

    def generate_conditioners(self,
                              protein: Protein,
                              vector: torch.Tensor,
                              magnitude: float) -> list[Conditioner]:
        """
        Combines substructure conditioning with the inflate conditioner to both
        remodel binder backbones and place them at a hotspot defined by the vector
        fed to inflate. Returns composed chroma conditioner object.
        """

        substructure = conditioners.SubstructureConditioner(
            protein = protein,
            backbone_model = self.model.backbone_network,
            selection = 'namesel receptor'
        ).to(self.device)

        displacement = conditioners.InflateConditioner(vector, magnitude).to(self.device)
        
        #if False:
        #    subsequence = conditioners.SubsequenceConditioner(
        #        design_model = self.model.design_network,
        #        protein = protein,
        #        selection = 'not namesel receptor',
        #        weight = 1.,
        #    ).to(self.device)
        #
        #    return [displacement, substructure, subsequence]
        
        return [displacement, substructure]

    @staticmethod
    def convert_cif_to_pdb(cif: PathLike) -> None:
        """
        Converts a cif file to pdb format using Biopython
        """
        parser = PDB.MMCIFParser(QUIET=True)
        structure = parser.get_structure('structure', str(cif))

        io = PDB.PDBIO()
        io.set_structure(structure)
        io.save(str(cif.with_suffix('.pdb')))
    
    @classmethod
    def from_toml(cls: Type[_T], 
                  config: PathLike) -> _T:
        config = tomllib.load(open(config, 'rb'))
        hotspots = config['hotspot']
        nn = config['neural_net']

        config = config['settings']
        input_dir = Path(config['input_path'])
        output_dir = Path(config['output_path'])
        try:
            hotspot_dir = Path(config['hotspot_path'])
        except KeyError:
            hotspot_dir = None
        
        binder_length = config['binder_length']

        n_rounds = config['num_rounds']
        n_backbones = config['num_backbones']
        n_designs = config['num_designs']
        diff_steps = config['diffusion_steps']
        store_in_memory = config['store_in_memory']
        memory_migration_freq = config['memory_migration_freq']

        model_weights = nn['design_weights']
        backbone_weights = nn['bb_weights']
        conditioner_weights = nn['cond_weights']
        device = nn['device']

        return cls(input_dir,
                   output_dir,
                   hotspot_dir,
                   hotspots,
                   binder_length,
                   n_rounds,
                   n_backbones,
                   n_designs,
                   diff_steps,
                   model_weights,
                   backbone_weights,
                   conditioner_weights,
                   device,
                   store_in_memory,
                   memory_migration_freq,
                  )

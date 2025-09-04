from molecular_simulations.analysis import Fingerprinter
from molecular_simulations.build import ImplicitSolvent
from molecular_simulations.simulate import Minimizer
from natsort import natsorted
import numpy as np
import os
from qcer import PeptideQualityControl
from pathlib import Path

tleap = '/lus/flare/projects/FoundEpidem/msinclair/envs/plinder/bin/tleap'
AMBERHOME='/lus/flare/projects/FoundEpidem/msinclair/envs/plinder'

try:
    print(os.environ['AMBERHOME'])
except KeyError:
    os.environ['AMBERHOME'] = AMBERHOME

intify = lambda x: int(''.join([c for c in x if c.isdigit()]))

def measure_energy(pdb: Path, system: list[str], exp_name: str) -> None:
    out_path = Path(f'energies/{exp_name}/fp')
    out_path.mkdir(exist_ok=True, parents=True)

    out_name = pdb.with_suffix('.npz').name
    
    intermediate = '_'.join(system) + '.pdb'
    minimized = 'min_' + '_'.join(system) + '.pdb'
    out = (path / intermediate).with_suffix('.prmtop')
    
    builder = ImplicitSolvent(None, 
                              original_pdb, 
                              use_amber=True, 
                              out=intermediate, 
                              tleap=tleap)
    builder.build()

    mini = Minimizer(out, 
                     out.with_suffix('.inpcrd'), 
                     out=minimized, 
                     platform='OpenCL',
                     device_ids=None)
    mini.minimize()

    fp = Fingerprinter(out, 
                       path / minimized, 
                       'resid 1 to 1365', 
                       'not resid 1 to 1365', 
                       out_path, 
                       out_name)
    fp.run()
    fp.save()

if __name__ == '__main__':
    from argparse import ArgumentParser
    from mpi4py import MPI

    parser = ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    _path = Path(args.path)

    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    size = COMM.Get_size()

    if rank == 0:
        calcs = []
        for rank in _path.glob('rank*'):
            _rank = intify(rank.name)
            for rnd in rank.glob('round*'):
                _round = intify(rnd.name)
                pdbs = natsorted(list(rnd.glob('*.pdb')))
                for pdb in pdbs:
                    if len(pdb.stem.split('_')) == 3:
                        inp_pdb, bb, design = pdb.stem.split('_')
                        calcs.append([str(x) for x in [_rank, _round, inp_pdb, bb, design]])

        calcs = np.array_split(calcs, size)

    else:
        calcs = None
    
    calcs = COMM.scatter(calcs, root=0)
    
    QC = PeptideQualityControl(
        max_repeat=4,
        max_appearance_ratio=0.5,
        max_charge=5,
        max_charge_ratio=0.5,
        max_hydrophobic_ratio=0.5,
        min_diversity=8,
    )
    out_path = Path('energies') / _path.name / 'fp'
    for calc in calcs:
        system = '_'.join(calc)
        if not (out_path / f'{system}.npz').exists():
            ra, ro, pdb, bb, de = calc 
            path = _path / f'rank{ra}' / f'round_{ro}'
            struc = '_'.join([pdb, bb, de])
        
            original_pdb = (path / struc).with_suffix('.pdb').resolve()

            if QC(str(original_pdb)):
                measure_energy(original_pdb, calc, _path.name)

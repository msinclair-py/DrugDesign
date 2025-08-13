from molecular_simulations.analysis import Fingerprinter
from molecular_simulations.build import ImplicitSolvent
from molecular_simulations.simulate import Minimizer
from natsort import natsorted
import numpy as np
import os
from pathlib import Path

tleap = '/lus/flare/projects/FoundEpidem/msinclair/envs/plinder/bin/tleap'
AMBERHOME='/lus/flare/projects/FoundEpidem/msinclair/envs/plinder'

try:
    print(os.environ['AMBERHOME'])
except KeyError:
    os.environ['AMBERHOME'] = AMBERHOME

intify = lambda x: int(''.join([c for c in x if c.isdigit()]))

def measure_energy(path: Path, system: list[str]) -> None:
    ra, ro, pdb, bb, de = system
    
    out_path = Path(f'energies/{path.name}/fp')
    out_path.mkdir(exist_ok=True, parents=True)

    out_name = '_'.join(system) + '.npz'
    
    if not (out_path / out_name).exists():
        path = path / f'rank{ra}' / f'round_{ro}'
        struc = '_'.join([pdb, bb, de])
        
        original_pdb = (path / struc).with_suffix('.pdb')
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

    path = Path(args.path)

    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    size = COMM.Get_size()

    if rank == 0:
        calcs = []
        for rank in path.glob('rank*'):
            _rank = intify(rank.name)
            for rnd in rank.glob('round*'):
                _round = intify(rnd.name)
                pdbs = natsorted(list(rnd.glob('*.pdb')))
                for pdb in pdbs:
                    if len(pdb.stem.split('_')) > 3:
                        continue

                    inp_pdb, bb, design = pdb.stem.split('_')
                    calcs.append([str(x) for x in [_rank, _round, inp_pdb, bb, design]])

        calcs = np.array_split(calcs, size)

    else:
        calcs = None
    
    calcs = COMM.scatter(calcs, root=0)
    
    out_path = Path('energies') / path.name / 'fp'
    for calc in calcs:
        if not (out_path / f'{calc}.npz').exists():
            measure_energy(path, calc)

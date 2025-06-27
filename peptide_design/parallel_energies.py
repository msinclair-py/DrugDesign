from molecular_simulations.analysis.interaction_energy import StaticInteractionEnergy
from molecular_simulations.build import ImplicitSolvent
from molecular_simulations.simulate import Minimizer
from mpi4py import MPI
from natsort import natsorted
import numpy as np
import os
from pathlib import Path
#import polars as pl

tleap = '/lus/flare/projects/FoundEpidem/msinclair/envs/plinder/bin/tleap'
AMBERHOME='/lus/flare/projects/FoundEpidem/msinclair/envs/plinder'

try:
    print(os.environ['AMBERHOME'])
except KeyError:
    os.environ['AMBERHOME'] = AMBERHOME

intify = lambda x: int(''.join([c for c in x if c.isdigit()]))

def measure_energy(ra, ro, ip, bb, de, pdb):
    out = f'{ra}_{ro}_{ip}_{bb}_{de}.pdb'
    builder = ImplicitSolvent(None, pdb, out=out, use_amber=True, tleap=tleap)
    builder.build()

    out = f'min_{ra}_{ro}_{ip}_{bb}_{de}.pdb'
    mini = Minimizer(builder.out, out)
    mini.minimize()

    sie = StaticInteractionEnergy(str(mini.out), 
                                  chain=' ', # AMBER is dumb
                                  platform='OpenCL', 
                                  first_residue=1, 
                                  last_residue=1365)
    sie.compute()

    return [ra, ro, ip, bb, de, sie.lj, sie.coulomb]

if __name__ == '__main__':
    #root = Path('peptides/whsc1_firstpass')
    #calcs = []
    #for rank in root.glob('*'):
    #    _rank = intify(rank.name)
    #    for rnd in rank.glob('*'):
    #        _round = intify(rnd.name)
    #        pdbs = natsorted(list(rnd.glob('*.pdb')))
    #        for pdb in pdbs:
    #            if len(pdb.stem.split('_')) > 3:
    #                continue
    #
    #            inp_pdb, bb, design = pdb.stem.split('_')

    #            calcs.append([_rank, _round, inp_pdb, bb, design, pdb])

    #print(calcs)
    #with open('energy_calcs.txt', 'w') as fout:
    #    for calc in calcs:
    #        fout.write(','.join([str(x) for x in calc]) + '\n')

    calcs = [line.strip().split(',') for line in open('energy_calcs.txt').readlines()]

    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()
    size = COMM.Get_size()

    calcs = np.array_split(calcs, size)[rank]
    
    out_file = Path(f'energies/rank{rank}.txt')
    if out_file.exists():
        results = [open(str(out_file), 'r').read().strip()]
    else:
        results = []

    for i, calc in enumerate(calcs):
        print(i, calc)
        results.append(','.join([str(x) for x in measure_energy(*calc)]))

        if i % 10 == 0:
            with open(str(out_file), 'w') as fout:
                fout.write('\n'.join(results))

    with open(str(out_file), 'w') as fout:
        fout.write('\n'.join(results))

#df = pl.DataFrame(results, schema={'rank': int, 
#                                   'round': int, 
#                                   'pdb': str, 
#                                   'backbone': int, 
#                                   'design': int, 
#                                   'lennard-jones': float, 
#                                   'coulombic': float})
#
#df.write_parquet('binder_energies.parquet')

#!/bin/bash
#PBS -N docking
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A FoundEpidem
#PBS -o logs/
#PBS -e logs/
#PBS -m abe

cd /eagle/projects/FoundEpidem/msinclair/ideals/ythdf1/docking
source /eagle/projects/FoundEpidem/docking/envs/docking_openeye/bin/activate

mpirun -n 40 -ppn 40 python dock.py -c config.toml -m True > logs/dock.log 2> logs/dock.err 

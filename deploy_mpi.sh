#!/bin/bash
#PBS -N chroma
#PBS -l select=30
#PBS -l filesystems=flare
#PBS -l walltime=6:00:00
#PBS -q prod
#PBS -A FoundEpidem
#PBS -e logs/
#PBS -o logs/
#PBS -M msinclair@anl.gov

cd /lus/flare/projects/FoundEpidem/msinclair/ideals/peptide_design

export NUM_NODES=30
export RANKS_PER_NODE=12
export TOTAL_NUM_RANKS=$((NUM_NODES * RANKS_PER_NODE))

source /lus/flare/projects/FoundEpidem/msinclair/envs/peptide_design/bin/activate
mpiexec -np $TOTAL_NUM_RANKS --ppn $RANKS_PER_NODE ./helpers/set_ze_mask.sh python mpi_chroma.py

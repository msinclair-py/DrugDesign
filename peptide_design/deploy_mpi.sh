#!/bin/bash
#PBS -N chroma
#PBS -l select=512
#PBS -l filesystems=flare
#PBS -l walltime=6:00:00
#PBS -q prod
#PBS -A FoundEpidem
#PBS -e logs/
#PBS -o logs/
#PBS -m abe
#PBS -M msinclair@anl.gov

cd /lus/flare/projects/FoundEpidem/msinclair/ideals/peptide_design

# To ensure GPU affinity mask matches the physical order of the GPUs on the node
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export CPU_AFFINITY="verbose,list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:48-55,152-159:56-63,160-167:64-71,168-175:72-79,176-183:80-87,184-191:88-95,192-199"
export CCL_LOG_LEVEL="DEBUG"
export NUMEXPR_MAX_THREADS=208
export ITEX_LIMIT_MEMORY_SIZE_IN_MB=8192
export ITEX_ENABLE_NEXTPLUGGABLE_DEVICE=0
# Set environment variable to print MPICH's process mapping to cores:
export HYDRA_TOPO_DEBUG=1

export RANKS_PER_NODE=12

mpiexec -np $(cat $PBS_NODEFILE | wc -l) -ppn 1 --pmi=pmix hostname > ./helpers/hostnamelist_diff.dat
export NUM_NODES="$(cat ./helpers/hostnamelist_diff.dat | wc -l)"

mpiexec -np $NUM_NODES -ppn 1 cp -r chroma_weights /dev/shm/
mpiexec -np $NUM_NODES -ppn 1 cp 2g3n.cif /tmp/

export TOTAL_NUM_RANKS=$((NUM_NODES * RANKS_PER_NODE))

source /lus/flare/projects/FoundEpidem/msinclair/envs/peptide_design/bin/activate
mpiexec -np $TOTAL_NUM_RANKS --ppn $RANKS_PER_NODE ./helpers/set_ze_mask.sh ./helpers/interposer.sh python mpi_chroma.py > logs/pep.log 2> logs/pep.err

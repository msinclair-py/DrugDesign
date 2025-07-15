def run_chroma(config, out_path):
    from designer import ChromaDesigner
    from pathlib import Path

    design = ChromaDesigner.from_toml(config)
    design.output_dir = design.output_dir / out_path
    design.run_rounds()

if __name__ == '__main__':
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    rank = COMM.Get_rank()

    config = 'config.toml'
    run_chroma(config, f'rank{rank}')

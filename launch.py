import os
import sys
import config

Nsim = sys.argv[1]

main_str = 'main.py'
job_str = 'IHDP'

path = 'Sim/Test' + str(config.Ntest)
if not os.path.exists(path):
    os.mkdir(path)

Ncores = 1  # Number of requested cores
Nhours = 32  # Requested walltime (hours)

for i in range(0, int(Nsim)):
    seed = i+1
    jobId = job_str + '_' + str(config.Ntest) + '_' + str(seed)
    slurmFileName = 'run_' + jobId + '.slurm'

    with open(slurmFileName, 'w') as fout:
        fout.write('#!/bin/bash -l\n')
        fout.write('\n')
        fout.write('#SBATCH --job-name ' + jobId + '\n')
        fout.write('#SBATCH --nodes 1\n')
        fout.write('#SBATCH --cpus-per-task ' + str(Ncores) + '\n')
        fout.write('#SBATCH --time ' + str(Nhours) + ':00:00\n')
        fout.write('#SBATCH --output ' + 'out/' + jobId + '.txt\n')
        fout.write('#SBATCH --mem=4000\n')
        # fout.write('#SBATCH --mail-type=FAIL\n')
        # fout.write('#SBATCH --mail-user=daolang.huang@aalto.fi\n')
        fout.write('\n')
        fout.write('python ' + main_str + ' ' + str(seed))

    os.popen('sbatch ' + slurmFileName)

for i in range(0, int(Nsim)):
    seed = i + 1
    jobId = job_str + '_' + str(config.Ntest) + '_' + str(seed)
    slurmFileName = 'run_' + jobId + '.slurm'
    os.remove(slurmFileName)
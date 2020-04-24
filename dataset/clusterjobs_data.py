import os
import itertools

HOME = '/gpfs01/bethge/home/pmamidanna/deep_proprioception/clusterjobs'

def write_header(file):
    file.write("#!/bin/bash\n")
    file.write("#PBS -j oe\n") # Merge stdout and stderr
    file.write("#PBS -l pmem=1000mb\n") # Set needed memory (per core!)
    file.write("#PBS -l nodes=1:ppn=1\n") # Set needed cores
    file.write("#PBS -l walltime=04:00:00\n") # Set walltime
    file.write("set -e\n") # Fail on errors
    file.write("export KMP_AFFINITY=none\n")
    file.write("export OMP_NUM_THREADS=1\n")

def write_command(file, job_id, label, plane):
    file.write('/gpfs01/bethge/home/pmamidanna/docker-deeplearning/agmb-docker-cpu ')
    file.write(f'run -i --name pm-job{job_id} --rm pranavm19/opensim:opensim-slim ')
    file.write('python3 /gpfs01/bethge/home/pmamidanna/deep_proprioception/code/')
    file.write(f'generate_data.py --label {label} --plane {plane} {job_id}')

def main():
    '''Generate cluster jobs to generate dataset.'''
    # os.mkdir(HOME)
    os.chdir(HOME)
    labels = range(1, 21)
    planes = ('vertical', 'horizontal')
    job_id = 20000

    for _ in range(75):
        for label in labels:
            job_id += 1
            job_name = f'job_gendata{job_id}.sh'
            print(job_name)
            with open(job_name, "w") as file:
                write_header(file)
                write_command(file, job_id, label, 'vertical')

if __name__=="__main__":
    main()

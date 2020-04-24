import os
import itertools

HOME = '/gpfs01/bethge/home/pmamidanna/deep_proprioception/clusterjobs'

def write_header(file, job_id, numprocs):
    file.write("#!/bin/bash\n")
    file.write("#PBS -j oe\n") # Merge stdout and stderr
    file.write("#PBS -l pmem=1000mb\n") # Set needed memory (per core!)
    file.write(f"#PBS -l nodes=1:ppn={numprocs}\n") # Set needed cores
    file.write("#PBS -l walltime=08:00:00\n") # Set walltime
    file.write("set -e\n") # Fail on errors
    file.write("export KMP_AFFINITY=none\n")
    file.write(f"export OMP_NUM_THREADS={numprocs}\n")
    file.write("myhostname=\"$(hostname)\" \n")
    file.write("if [ \"$myhostname\" == \"cn65\" ]; then \n")
    file.write("\tsleep 20000 \nfi \n")
    file.write(f"hostname &> /gpfs01/bethge/home/pmamidanna/logs/log{job_id}.txt\n \n")

def write_command(file, job_id, c1, c2, intype, var_type):
    file.write('/gpfs01/bethge/home/pmamidanna/docker-deeplearning/agmb-docker-cpu ')
    file.write(f'run -i --name pm-jobsvm{job_id} --rm pranavm19/opensim:opensim-slim ')
    file.write('python3 /gpfs01/bethge/home/pmamidanna/deep_proprioception/code/')
    file.write(f'binarysvm.py --class1 {c1} --class2 {c2} --{var_type} --input_type {intype} ')
    file.write(f'&>> /gpfs01/bethge/home/pmamidanna/logs/log{job_id}.txt\n')

def main():
    '''Generate cluster jobs to generate dataset.'''
    os.chdir(HOME)
    labels = range(1, 21)
    input_type = ['endeffector_coords', 'joint_coords', 'muscle_coords', 'spindle_firing']
    var_types = ['full_variability', 'suppressed']
    job_id = 0

    for (c1, c2), intype, var_type in itertools.product(
            list(itertools.combinations(labels, 2)), input_type, var_types):
        job_id += 1
        job_name = f'job_svm_{intype}_{var_type}_{c1}_{c2}.sh'
        if intype == 'endeffector_coords' or intype == 'joint_coords':
            numprocs = 4
        else:
            numprocs = 8
        
        with open(job_name, "w") as file:
            write_header(file, job_id, numprocs)
            write_command(file, job_id, c1, c2, intype, var_type)

if __name__=="__main__":
    main()

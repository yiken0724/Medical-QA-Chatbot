#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=10           # Number of CPU to request for the job
#SBATCH --mem=32GB                 # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=00-12:00:00         # How long to run the job for? Jobs exceed this time will be terminated
                                # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=/common/home/projectgrps/CS425/CS425G9/v9-%j.out          # Where should the log files go?
                                # You must provide an absolute path eg /common/home/module/username/
                                # If no paths are provided, the output file will be placed in your current working directory
#SBATCH --requeue                   # Remove if you do not want the workload scheduler to requeue your job after preemption

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=project                 # The partition you've been assigned
#SBATCH --account=cs425   # The account you've been assigned (normally student)
#SBATCH --qos=cs425qos       # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=nicolastang.2022@scis.smu.edu.sg,cheryl.loh.2023@scis.smu.edu.sg,klement.goh.2021@scis.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=lora-tune-v9     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information
module purge
module load Python/3.11.7
module load CUDA/12.4.0

# Create a virtual environment can be commented off if you already have a virtual environment
# python3.11 -m venv ~/lora-venv

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
source ~/lora-venv/bin/activate

# If you require any packages, install it as usual before the srun job submission.
pip3 install pandas
pip3 install torch transformers datasets peft hf-xet

# Submit your job to the cluster
srun --gres=gpu:1 python ./train.py

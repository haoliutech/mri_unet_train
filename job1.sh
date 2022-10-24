#!/bin/bash
#export HOME=/hpcfs/users/a1787848
#SBATCH -p batch        	                                # partition (this is the queue your job will be added to) 
#SBATCH -N 1               	                                # number of nodes (use a single node)
#SBATCH -n 1              	                                # number of cores (sequential job => uses 1 core)
#SBATCH --time=01:00:00    	                                # time allocation, which has the format (D-HH:MM:SS), here set to 1 hour
#SBATCH --mem=4GB         	                                # specify the memory required per node (here set to 4 GB)

#module load Python/3.8.6

# Configure notifications 
#SBATCH --mail-type=END                                         # Send a notification email when the job is done (=END)
#SBATCH --mail-type=FAIL                                        # Send a notification email when the job fails (=FAIL)
#SBATCH --mail-user=a1787848@student.adelaide.edu.au          # Email to which notifications will be sent

# Execute your script (due to sequential nature, please select proper compiler as your script corresponds to)
export HOME=/hpcfs/users/a1787848/
module load Python/3.8.6
pip install -r requirements.txt
python run_no_wandb.py
#/bin/bash
 
#PBS -N 73T-130R
#PBS -l select=1:ncpus=24:mem=1003gb,walltime=168:00:00
#PBS -q ieng
#PBS -M msiddig@clemson.edu

module add gnu-parallel
module add julia/0.6.2 


cd $PBS_O_WORKDIR

#cat $PBS_NODEFILE > test_nodes.txt

parallel < commands.txt

#rm ./test_nodes.txt

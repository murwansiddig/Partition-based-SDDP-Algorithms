#/bin/bash
 
#PBS -N VLVA_QP
#PBS -l select=1:ncpus=24:mem=1000gb,walltime=72:00:00
#PBS -q bigmem
#PBS -M msiddig@clemson.edu

module add gnu-parallel
module add julia/0.6.2-gcc
module add gurobi


cd $PBS_O_WORKDIR

#cat $PBS_NODEFILE > test_nodes.txt

parallel < commands.txt

#rm ./test_nodes.txt

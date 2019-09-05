#!/bin/bash

file="./paramsList.txt"
ofile="./commands.txt"
runLog="./run.log"

rm -f $ofile

cat $file | while read line || [ -n "$line" ]; do
    fname="runLog.$(echo $line | tr -d [:space:]).txt"
    echo "echo $(date "+%Y-%m-%d %H:%M:%S") > $fname; echo Params: $line >> $fname; julia MAIN.jl $line >> $fname" >> $ofile
done

qsub ./jobSubmit.sh

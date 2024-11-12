#!/bin/bash

# The PID of the existing background process (replace 12345 with the actual PID)
target_pid=314248

# # Loop until the process with PID $target_pid is no longer running
while ps -p $target_pid > /dev/null; do
    echo "Waiting for process $target_pid to finish..."
    sleep 10  # Wait for 1 second before checking again
done

# Once the existing background process finishes, continue with the script
echo "Process $target_pid has finished. Continuing with the script..."

# echo "Starting Benchmark 1"
nohup python3 benchmark.py > "fetaqa_2500_clength.out"
# wait

# export CUDA_VISIBLE_DEVICES=3

# for i in {1..10}; do
#   echo "Starting Benchmark 2 625 Query ${i}"
#   nohup python3 benchmark2.py > "fetaqa_625_bge_query${i}.out"
#   wait
# done
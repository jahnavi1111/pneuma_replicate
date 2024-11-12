#!/bin/bash

# The PID of the existing background process (replace 12345 with the actual PID)
target_pid=2057770

# # Loop until the process with PID $target_pid is no longer running
while ps -p $target_pid > /dev/null; do
    echo "Waiting for process $target_pid to finish..."
    sleep 10  # Wait for 10 second before checking again
done

# Once the existing background process finishes, continue with the script
echo "Process $target_pid has finished. Continuing with the script..."

# echo "Starting Benchmark 1"
# nohup python3 benchmark.py > "fetaqa_5000_clength.out"
# wait

export CUDA_VISIBLE_DEVICES=0

for i in {1..10}; do
  echo "Starting Benchmark 2 625 Query ${i}"
  nohup python3 benchmark2_625.py > "fetaqa_625_load_query${i}.out"
  wait
done
# for i in {1..10}; do
#   echo "Starting Benchmark 2 1250 Query ${i}"
#   nohup python3 benchmark2_1250.py > "fetaqa_1250_load_query${i}.out"
#   wait
# done
# for i in {1..10}; do
#   echo "Starting Benchmark 2 2500 Query ${i}"
#   nohup python3 benchmark2_2500.py > "fetaqa_2500_load_query${i}.out"
#   wait
# done
# for i in {1..10}; do
#   echo "Starting Benchmark 2 5000 Query ${i}"
#   nohup python3 benchmark2_5000.py > "fetaqa_5000_load_query${i}.out"
#   wait
# done
# for i in {1..10}; do
#   echo "Starting Benchmark 2 10330 Query ${i}"
#   nohup python3 benchmark2_10330.py > "fetaqa_10330_load_query${i}.out"
#   wait
# done

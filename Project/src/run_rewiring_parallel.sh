#!/bin/bash

LISTS=(10 20 30 40 50 60 70 80 90)
ITERATIONS=(0 1 2 3 4 5 6 7 8 9)
MAX_PARALLEL_JOBS=10  # Adjust the number of parallel jobs

# Function to run the command
run_command() {
    list=$1
    iteration=$2
    python main.py "1000" "../data/rewiring_overlap_data/list_multiplex_0.$list.txt" "../output_files/rewiring_overlap/output_file$list/iteration_$iteration.txt"
}
export -f run_command

# Generate and run the commands in parallel
parallel -j $MAX_PARALLEL_JOBS run_command ::: "${LISTS[@]}" ::: "${ITERATIONS[@]}"
#!/bin/bash

OVERLAPS=(0 0.20 0.40)
LISTS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
ITERATIONS=(0 1 2 3 4 5)
MAX_PARALLEL_JOBS=10  # Adjust the number of parallel jobs

# Function to run the command
run_command() {
    overlap=$1
    list=$2
    iteration=$3
    python main.py "1000" "../data/multiplex_overlap_$overlap/list_multiplex_$list.txt" "../output_files/output_file_$overlap/list_$list/iteration_$iteration.txt"
}
export -f run_command

# Generate and run the commands in parallel
parallel -j $MAX_PARALLEL_JOBS run_command ::: "${OVERLAPS[@]}" ::: "${LISTS[@]}" ::: "${ITERATIONS[@]}"

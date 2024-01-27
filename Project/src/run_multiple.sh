#!/bin/bash

ARGS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

# Loop through the arguments and run the program

for i in "${ARGS[@]}"
do
    python main.py "1000" "../data/multiplex_overlap_0/list_multiplex_$i.txt" "../output_files/output_file_0_multiple_$i.txt"
done

for i in "${ARGS[@]}"
do
    python main.py "1000" "../data/multiplex_overlap_0.40/list_multiplex_$i.txt" "../output_files/output_file_04_multiple_$i.txt"
done

for i in "${ARGS[@]}"
do
    python main.py "1000" "../data/multiplex_overlap_0.20/list_multiplex_$i.txt" "../output_files/output_file_02_multiple_$i.txt"
done
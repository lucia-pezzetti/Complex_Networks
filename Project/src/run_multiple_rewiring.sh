#!/bin/bash

ARGS=(10 20 30 40 50 60 70 80 90)

# Loop through the arguments and run the program
for i in "${ARGS[@]}"
do
    python main.py "1000" "../data/rewiring_overlap_data/list_multiplex_0.$i.txt" "../output_files/rewiring_overlap/output_file.txt"
done
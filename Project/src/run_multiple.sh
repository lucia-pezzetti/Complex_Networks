#!/bin/bash

ARGS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

# Loop through the arguments and run the program
for i in "${ARGS[@]}"
do
    python main.py "1000" "../data/multiplex_overlap_0.40_1/list_multiplex_$i.txt"
done

for i in "${ARGS[@]}"
do
    python main.py "1000" "../data/multiplex_overlap_0.20_1/list_multiplex_$i.txt"
done

for i in "${ARGS[@]}"
do
    python main.py "1000" "../data/multiplex_rho_1/list_multiplex_$i.txt"
done
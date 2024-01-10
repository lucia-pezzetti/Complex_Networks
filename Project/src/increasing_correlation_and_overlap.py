import os
import subprocess
import sys

# Check for correct number of arguments
if len(sys.argv) <= 3:
    print("Insert <layer1> <layer2> <iteration_number> as input! Exit now")
    sys.exit(1)

L1 = sys.argv[1]
L2 = sys.argv[2]
iteration_number = sys.argv[3]

# Tuning the rho correlation from -1 to 1
subprocess.run(["sh", "tune_rho_adiab.sh", L1, L2, "0", "0.1", "1", "0.005", "0.0000001", iteration_number])

# For each of the duplex with a given correlation value, we increase the overlap to 0.2 and 0.4
counter = 1

for i in [x * 0.1 for x in range(-10, 11)]:
    if counter == 11:
        i = 0.0
    i_formatted = "{:.1f}".format(i)
    with open("list_multiplex.txt", "w") as outfile:
        subprocess.run(["ls", "-v", f"{L1}_rho_{i_formatted}", f"{L2}_rho_{i_formatted}"], stdout=outfile)
    with open("nodes_unique.txt", "w") as outfile:
        subprocess.run(["cat", f"{L1}_rho_{i_formatted}", f"{L2}_rho_{i_formatted}"], stdout=outfile)
        subprocess.run(["awk", "{ print $1 \"\\n\" $2 }", "|", "sort", "-n", "|", "uniq"], stdout=outfile)
    
    # Rewiring the duplex using biased edge rewiring
    subprocess.run(["./Rewiring_overlap/tune_overlap", "list_multiplex.txt", "nodes_unique.txt", "I", "-m", "0.4"])
    
    # Renaming the file with overlap and correlation
    for overlap in [x * 0.2 for x in range(3)]:
        overlap_formatted = "{:.2f}".format(overlap)
        #os.rename(f"{L1}_{overlap_formatted}", f"{L1}_overlap_{overlap_formatted}_rho_{i_formatted}")
        #os.rename(f"{L2}_{overlap_formatted}", f"{L2}_overlap_{overlap_formatted}_rho_{i_formatted}")
    
    counter += 1

# Moving files to the proper directories
for overlap in [x * 0.2 for x in range(3)]:
    overlap_formatted = "{:.2f}".format(overlap)
    if overlap == 0.0:
        os.mkdir(f"multiplex_rho_{iteration_number}")
        for rho in [x * 0.1 for x in range(-10, 11)]:
            rho_formatted = "{:.1f}".format(rho)
            if rho == -0.0:
                rho_formatted = "0.0"
            os.rename(f"{L1}_rho_{rho_formatted}", f"multiplex_rho_{iteration_number}")
            os.rename(f"{L2}_rho_{rho_formatted}", f"multiplex_rho_{iteration_number}")

        os.chdir(f"multiplex_rho_{iteration_number}")
        counter = 0
        for rho in [x * 0.1 for x in range(-10, 11)]:
            rho_formatted = "{:.1f}".format(rho)
            if rho == -0.0:
                rho_formatted = "0.0"
            with open(f"list_multiplex_{counter}.txt", "w") as outfile:
                subprocess.run(["ls", "-v", f"{L1}_rho_{rho_formatted}", f"{L2}_rho_{rho_formatted}"], stdout=outfile)
            counter += 1
        os.chdir("..")
    else:
        os.mkdir(f"multiplex_overlap_{overlap_formatted}_{iteration_number}")
        for rho in [x * 0.1 for x in range(-10, 11)]:
            rho_formatted = "{:.1f}".format(rho)
            if rho == -0.0:
                rho_formatted = "0.0"
            os.rename(f"{L1}_overlap_{overlap_formatted}_rho_{rho_formatted}", f"multiplex_overlap_{overlap_formatted}_{iteration_number}")
            os.rename(f"{L2}_overlap_{overlap_formatted}_rho_{rho_formatted}", f"multiplex_overlap_{overlap_formatted}_{iteration_number}")

        os.chdir(f"multiplex_overlap_{overlap_formatted}_{iteration_number}")
        counter = 0
        for rho in [x * 0.1 for x in range(-10, 11)]:
            rho_formatted = "{:.1f}".format(rho)
            if rho == -0.0:
                rho_formatted = "0.0"
            with open(f"list_multiplex_{counter}.txt", "w") as outfile:
                subprocess.run(["ls", "-v", f"{L1}_overlap_{overlap_formatted}_rho_{rho_formatted}", f"{L2}_overlap_{overlap_formatted}_rho_{rho_formatted}"], stdout=outfile)
            counter += 1
        os.chdir("..")

# Cleanup
for file in os.listdir("."):
    if file.endswith("_bottom_to_top"):
        os.remove(file)

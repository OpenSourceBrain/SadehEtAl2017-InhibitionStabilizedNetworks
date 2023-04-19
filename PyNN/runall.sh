#!/bin/bash
set -e
for i in `seq 100 100 4000`;
do
    echo "==============================="
    echo "Running with seed: "$i

    python runNetwork.py nest .75 1000  $i &&  python  analysis_perturbation_pynn.py .75 1000 -nogui -average -legend

done 

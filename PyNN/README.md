### PyNN Model

This version of the network can run in PyNN using simulator NEST

To simulate the activity of 2000 cells (1600 excitatory in red, 400 inhibitory in blue with 0.75 of the inhibitory cells (darker blue) perturbed), and then plot result:

```
python runNetwork.py nest .75 2000
python analysis_perturbation_pynn.py .75 2000
```

![0.75](test_0.75.png)

To simulate activity (2000 cells total, with 0.1 of the inhibitory cells perturbed), and then plot result:

```
python runNetwork.py nest .1 2000
python analysis_perturbation_pynn.py .1 2000
```

![0.1](test_0.1.png)

The script [runall.sh](runall.sh) will run 40 simulations and plot the average of these:


![40](rates.png)




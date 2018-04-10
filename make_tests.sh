# Make smaller/quicker network for testing (e.g. on Travis)
sed -i -e s/'set_total_population_size(2000)'/'set_total_population_size(50)'/g SpikingSimulationModels/defaultParams.py  # Make smaller network size
sed -i -e s/'Ntrials = 5'/'Ntrials = 2'/g SpikingSimulationModels/defaultParams.py  # Make fewer trials

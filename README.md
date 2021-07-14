## boltzmann-generator-playground

This is a playground of [Boltzmann Generator](https://science.sciencemag.org/content/365/6457/eaaw1147) codes. Here is the plan:

[x] Explore the BG sampling model with PDZ3 datasets (See pdz3/PG_pdz3_TF2.ipynb)

[x] Provide a more general workflow and preprocess scripts for using this framework in other proteins. (See pdz3/PG_pdz3_TF2.ipynb)

[x] Upgrade to Tensorflow 2 

Original Codes and notebooks in `example/` folder are knidly provided by the authors of Boltzmann Generator paper via [Zenodo](https://zenodo.org/record/3242635#.YIgr931KhTY) an emails.

Above targets are all accomplished. Next possible TODOs:

1. Remove unnecessary legacy codes in the current TF2 BG scripts

2. Remove unncessary playground jupyter notebooks 

How to use TF 2.0 BG: 

1. `git clone https://github.com/minhuanli/boltzmann-generator-playground.git`

2. `cd examples/deep_boltzmann`

3. `pip install requirements.txt`

4. `python setup.py install`

Then you should be ready to play with the example notebook in `playground/pdz3/PG_pdz3_TF2.ipynb`
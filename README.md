## boltzmann-generator-playground

This is a repository of [Boltzmann Generator](https://science.sciencemag.org/content/365/6457/eaaw1147) models. I made several modifications:

1. Upgrade to support tensorflow 2.

2. Complete the list in protein internal coordination calculations. Now it should support most proteins (except some with small molecules).

3. Add extra optimization target to accomodate the crystallography data, with help of the differentiable Structural Factor calculator in my other project DeepRefine (sorry, not published yet).

Original Codes and notebooks are knidly provided by the authors of Boltzmann Generator paper via [Zenodo](https://zenodo.org/record/3242635#.YIgr931KhTY) an emails.

How to use:

1. `git clone https://github.com/minhuanli/boltzmann-generator-playground.git`

3. `pip install requirements.txt`

4. `python setup.py install`

Then you should be ready to play with the example notebook in `examples/pdz3/PDZ3_BGTraining.ipynb`
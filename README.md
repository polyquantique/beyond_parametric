# Beyond the Parametric Approximation:
This code can be used to reproduce results in https://arxiv.org/abs/2312.09239. <br>

## Code Organization:
The repository contains two folders: `full_numerical_solution` and `cumulant_expansion_method`.
- In the `full_numerical_solution` folder, we have python code in the file `pdc_functions.py`, which has all the functions that can be used to analyze population, squeezing, the zero delay autocorrelation function, entanglement and the marginal probability distributions of the modes. In this folder, we also have provided a jupyter notebook `example.ipynb` that shows how to compute these quantities, and the following functions are essential in order to perform this analysis:
  1. `state_evolution`: this function outputs the state of the system as a function of time.
  2. `moments`: this function can compute arbitrary moments of the form for a given input state. These moments can then be used to compute populations, variances and zero delay autocorrelation function of the various modes.
  3. `pump_mat_purity`, `signal_mat_purity`: these functions can be used to obtain purity of the pump and the signal modes after tracing out the other modes of the system.
  4. `witness_fourth_order`, `witness_sixth_order`: these functions can be used to obtained the value of the witness functions that detect entanglement in this system.
  5. `pump_marg_prob`, `signal_marg_prob`: these functions can be used for computing the photon statistics of the pump and the signal mode.

- In the folder `cumulant_expansion_method`, we have Julia code which uses the `QuantumCumulants` package to compute various quantities of interest using the cumulant expansion method. This folder has four different jupyter notebooks that computes various quantities of interest using the second, third, fourth and fifth order cumulant expansion method.


 
## Citing:
If you find this code useful in your research, please consider citing our paper:

```bib
@article{chinni2023beyond,
  title={Beyond the parametric approximation: pump depletion, entanglement and squeezing in macroscopic down-conversion},
  author={Chinni, Karthik and Quesada, Nicol{\'a}s},
  journal={arXiv preprint arXiv:2312.09239},
  year={2023}
}
```

## Funding:
Funding for our work has been provided by
* Ministère de l'Économie et de l’Innovation du Québec, 
* Natural Sciences and Engineering Research Council of Canada,
* Fonds de recherche du Québec-Nature et technologies (FRQNT) under the Programme PBEEE / Bourses de stage postdoctoral

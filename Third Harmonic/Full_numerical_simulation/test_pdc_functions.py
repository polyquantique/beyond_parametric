"""
This script will do sanity checks on the functions defined in the module:
    pdc_functions 
"""
import pytest
import numpy as np
from pdc_functions import pop, rel_dist_ind, state_evolution, moments

# pylint: disable=too-many-locals


@pytest.mark.parametrize("mean", [10, 30, 100])
def test_rel_dist_ind(mean):
    """
    Checks if the indices produced for truncating coherent state are correct
    """
    # Identifying appropriate values of n in the Poisson dist. based on the function
    ind = rel_dist_ind(mean)

    assert pop(ind[-1] + 1, mean) < 10**-16
    assert pop(ind[-1], mean) > 10**-16


@pytest.mark.parametrize("alp_sq", [10, 20, 30])
def test_moments(alp_sq):
    """
    Checks if
    1. <alpha,0,0|ap^{dag}ap|alpha,0,0> = alpha_0^{2}
    2. <alpha,0,0|as^{dag}as|alpha,0,0> = 0
    3. <alpha,0,0|ai^{dag}ai|alpha,0,0> = 0
    """
    # Identifying appropriate values of n in the Poisson dist.
    n_arr = rel_dist_ind(alp_sq, 10 ** (-16))

    hil_spa_dim = np.sum(n_arr + 1)  # ll is the Hilbert space dimension
    psi = np.zeros(hil_spa_dim)
    ind1 = 0
    for dummy, n_val in enumerate(n_arr):
        psi[ind1] = np.sqrt(pop(n_val, alp_sq))
        ind1 = ind1 + n_val + 1

    assert np.isclose(moments(1, 1, 0, 0, 0, 0, psi, n_arr[0], n_arr[-1]), alp_sq)
    assert np.isclose(moments(0, 0, 1, 1, 0, 0, psi, n_arr[0], n_arr[-1]), 0)
    assert np.isclose(moments(0, 0, 0, 0, 1, 1, psi, n_arr[0], n_arr[-1]), 0)


@pytest.mark.parametrize("alp_sq", [10])
def test_state_evolution(alp_sq):
    """
    Checks if output states are all normalized, conserved quantities remain
    constant with time
    """

    # Threshold for the calculations:
    prob_ths = 10 ** (-16)

    tim_end = 0.5
    delt_t = 0.1

    output_data = state_evolution(alp_sq, tim_end, delt_t, prob_ths)

    output_states = output_data[
        1::, 1::
    ]  # state as a function of time is stored in rows for each time
    time_arr = output_data[1::, 0]  # times values of the state in each row
    t_len = len(time_arr)  # length of the time value array

    num_arr = output_data[0, 1:]  # number of excitations stored along each column

    ##n1 and n2 accounts for truncation of the Hilbert space as it
    ## #appears in psi(t)=sum_{n=n1}^{n2} sum_{k=0}^{n} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i} )
    n1_val = int(num_arr[0])
    n2_val = int(num_arr[-1])

    cons_qtty1 = np.zeros(
        t_len
    )  # Will store pump photons as a function of time in this array
    cons_qtty2 = np.zeros(
        t_len
    )  # Will store signal photons as a function of time in this array
    for tim_ind in range(0, t_len, 1):
        cons_qtty1[tim_ind] = moments(
            1, 1, 0, 0, 0, 0, output_states[tim_ind, :], n1_val, n2_val
        ) + moments(0, 0, 1, 1, 0, 0, output_states[tim_ind, :], n1_val, n2_val)
        cons_qtty2[tim_ind] = moments(
            0, 0, 1, 1, 0, 0, output_states[tim_ind, :], n1_val, n2_val
        ) - moments(0, 0, 0, 0, 1, 1, output_states[tim_ind, :], n1_val, n2_val)

    assert np.allclose(cons_qtty1, np.repeat(alp_sq, t_len))
    assert np.allclose(cons_qtty2, np.zeros(t_len))
    assert np.allclose(np.linalg.norm(output_states, axis=1), np.ones(t_len))

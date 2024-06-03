"""
This program is useful for computing the following:
    1. "state_evolution" function computes the state of the system in the
        PDC process as a function of time. 
    
    2. "moments" function can compute arbitrary moments of the state, which
       can then be used to compute populations, variances of the quadratures,
       zero delay autocorrelation functions of the various modes.
    
    3. "pump_mat_purity" and "sig_mat_purity" functions can be used to compute
       the purity of the pump or signal mode (after tracing out other modes of 
       the system).
    
    4. "witness_fourth_order" and "witness_sixth_order" functions can be used
        to compute the values of the entanglement witness.
       
    5. "pump_marg_prob" and "signal_marg_prob" functions can be used to 
        compute the photon statistics of the pump and the signal mode.
       
"""
import numpy as np
from scipy.stats import poisson
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from numba import jit

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

def pop(num, alp_sq):
    """
    Parameters
    ----------
    num : non-negative integer
    alp_sq : non-negative integer, mean and variance of the distribution.

    Returns
    -------
    returns Poisson probability Pr(x=num) with mean alp_sq
    """
    return poisson.pmf(num, alp_sq)


def rel_dist_ind(alp_sq, pb_th=10 ** (-16)):
    """
    Parameters
    ----------
    alp_sq : non-negative real number, mean and variance of the distribution.
    pb_th : non-negative real number,threshold for identifying the values of n
        whose probability is greater than pb_th

    Returns
    -------
    relevant_indices : the values of n whose Poisson prob(n,mean=alp_sq)>pb_th

    """
    n_arr = np.arange(0, 11 * alp_sq, 1)  # n goes from 0 to mean+10 times variance
    dist = pop(n_arr, alp_sq)  # probability distribution for n_arr values

    # np.flatnonzero(arg) gives indices of the elements where the arg is True
    # (and nonzero, in general)
    relevant_indices = np.flatnonzero(
        dist > pb_th
    )  # n values that satisfy the threshold
    return relevant_indices


def hami_elements(num_pump_max, ind_nu):
    """
    Parameters
    ----------
    num_pump_max : non-negative integer, the maximum number of pump photons
                  allowed in the subspace, whose Hamiltonian elements are
                  outputted from this function.
    ind_nu : non-negative integer, index ranging from 0 to num_pump_max-1
             (these indices correspond to lower-diagonal elements).

    Returns
    -------
    returns real number, lower diagonal elements of the Hamiltonian in the
            subspace defined by num_pump_max corresponding to ind_nu of the
            matrix with size, num_pump_max + 1
    """
    # See the paragraph between Eq. (6)  and Eq. (7) in the manuscript for
    # more details on this

    return np.sqrt(num_pump_max - ind_nu) * (ind_nu + 1)


def state_evolution(alp_sq, t_end, delt_t, pb_th, parts=1, in_arr=0):
    """
    Parameters
    ----------
    alp_sq : non-negative real number, alpha square indicating the average
             number of pump photons at time=0.

    t_end : non-negative real number, end time of the evolution.
            (Numbers relevant for our analysis (to help guide the user):
            t_end=0.5 for alp_sq=100
            t_end=0.15 for alp_sq=1000
            t_end=0.06 for alp_sq=10000)

    dt : non-negative real number, time step of the evolution.
        (Numbers reasonable for our analysis (to help guide the user):
        dt=0.01 for alp_sq=100
        dt=0.001 for alp_sq=1000
        dt=0.0001 for alp_sq=10000)

    pb_th : non-negative real number. It is the probability threshold that is
            used for truncating the Hilbert space. It cuts off the basis
            elements whose associated probability is smaller than pb_th at
            initial time.

    parts : non-negative integer, used for parallel programming.
            Tells how many parts you want to break the evolution.
            For no parallel programming, choose parts=1 (set as default).

    in_arr : non-negative integer, used for parallel programming.
        This value could come from SLURM script when parallel programming is
        used. It tells the computer to run "in_arr" part of the "parts" code.
        For instance if one chooses parts=5,in_arr ranges between 0 and 4
        and SLURM runs all these pieces simulatenously in parallel!
        Choose in_arr=0 when no parallel programming is used (set as default).

    Returns
    -------
    dat: state coefficients as a function of time in the number basis
    dat is arranged in the following manner:
    ###############################################################################
    Data is aranged such that we have states at different times arranged along
    different rows(besides the first row).The first row just contains the value
    of Np+NS corresponding to the state coefficients.
    More specifically, the output data will appear as follows:
     ________________________________________________________________________________
    | 0|   n1      |    n1       |...|   n1     ||...||n2(=n1+r)  |...|   n2       ||
    |__|___________|_____________|___|__________||___||___________|___|____________||
    |t1| b(n1,0,t1)| b(n1-1,1,t1)|...|b(0,n1,t1)||...||b(n2,0,t1) |...|b(0,n2+1,t1)||
    |t2| b(n1,0,t2)| b(n1-1,1,t2)|...|b(0,n1,t2)||...||b(n2,0,t2) |...|b(0,n2+1,t2)||
    |: |     :     |     :       |...|    :     ||...||    :      |...|     :      ||
    |: |     :     |     :       |...|    :     ||...||    :      |...|     :      ||
    |tn| b(n1,0,tn)| b(n1-1,1,tn)|...|b(0,n1,tn)||...|||b(n2,0,tn)|...|b(0,n2+1,tn)||
    |__|___________|_____________|___|__________||___||___________|___|____________||
    where b(n,k,t)=<n-k,k,k|psi(t)>. Note that b(n,k,t) = \beta_{n-k,k}(t) as
    defined in Eq. (7b) of the manuscript, n1 and n2 accounts for truncation of
    the Hilbert space as it appears in
    psi(t)=sum_{n=n1}^{n2} sum_{k=0}^{n} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i} )
    ###############################################################################
    """
    ##times values of interest for state evolution:
    t_arr = np.linspace(start=0, stop=t_end, num=int(t_end / delt_t) + 1, endpoint=True)

    # Identifying appropriate values of n in the Poisson dist.:
    n_arr = rel_dist_ind(alp_sq, pb_th)
    ##########################################################################
    # Splits the values of n_arr (useful for parallel programming only):
    n_arr11 = np.array_split(n_arr.astype(int), parts)
    n_arr = n_arr11[in_arr]  # n_arr of interest for this program is then used
    ############################################################################
    hil_spa_dim = np.sum(n_arr + 1)  # ll is the Hilbert space dimension

    # Stats array is used to store the state for various times:
    stats = np.zeros([len(t_arr), hil_spa_dim])
    nn_arr = np.zeros(hil_spa_dim)

    l_temp2 = 0
    for dummy, n_val in enumerate(n_arr):
        # print("n_running:", n_val, end="\r", flush=True)
        weight_n = pop(n_val, alp_sq)  # Poisson probability P(N)

        arr = np.arange(0, n_val, 1)
        low_diag = hami_elements(n_val, arr)
        subsupdiagonals = [low_diag, -low_diag]

        ini_state = np.zeros(n_val + 1)
        ini_state[0] = 1

        # Hamiltonian in the subspace:
        hami = diags(subsupdiagonals, [-1, 1])

        # l_temp1 and l_temp2 determine where |psi_{n_val}> (see Eq.7a in the
        # manuscript)   is stored
        l_temp1 = l_temp2
        l_temp2 = n_val + 1 + l_temp2

        # For more details on expm_multiply see,
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html

        # See Eq (7a) in the manuscript for more details:
        stats[:, l_temp1:l_temp2] = np.sqrt(weight_n) * expm_multiply(
            hami,
            ini_state,
            start=0,
            stop=t_end,
            num=int(t_end / delt_t) + 1,
            endpoint=True,
            traceA=0,
        )

        # nn_arr when placed over stats[tt,:] will have Np+Ns corresponding to
        # the state coefficients
        nn_arr[l_temp1:l_temp2] = np.repeat(n_val, n_val + 1)

    nn_arr = np.append(0, nn_arr)  # Produces 1st row of the output
    data = np.column_stack(
        (t_arr, stats)
    )  # States as a func of time arranged next to t_arr
    dat = np.vstack(
        (nn_arr, data)
    )  # Stacks first row with the matrix containing states

    return dat


@jit(nopython=True, cache=True)
def perm(num, k):
    """
    Parameters
    ----------
    num : nonnegative integer.
    k : nonnegative integer.

    Returns
    -------
    ans : nonnegative integer
        returns the value of P(num,k)=num!/(num-k)!

    """
    ans = 1.0
    for i in range(num - k + 1, num + 1):
        ans *= i  # P(num,k)=num*(num-1)*...*(num-k+1)
    return ans


@jit(nopython=True, cache=True)
def coef(num_pmp, num_sig, num_id, psi, n_start, n_end):
    """
    Parameters
    ----------
    num_pmp (pump photons): integer (ranging between 0 and n_end for nonzero value)
    num_sig (signal photons): integer (ranging between 0 and n_end for nonzero value)
    num_id (idler photons): integer (ranging between 0 and n_end for nonzero value)
    psi: 1d array with real numbers (state of the system)

    n_start: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i};
     see Eq. (7b) in the manuscript)

    n_end: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i} )

    Returns
    -------
    returns real number, state coefficient <np,ns,ni|psi(time)>

    """
    if num_pmp < 0 or num_sig < 0 or num_id < 0:  ## unphysical values
        return 0

    if (
        num_pmp > n_end or num_sig > n_end or num_id > n_end
    ):  ## truncation of Hilbert space
        return 0

    if num_sig != num_id:  ## selection rules
        return 0

    extns = num_pmp + num_sig  ## extns: excitations
    if n_start <= extns <= n_end:
        # accounts for state vector truncation n1<= n<=n2

        # exc_ind is the number of elements in the state vector before "extns"
        # excitations (pump+idler photons) given the way the state
        # coefficients are arranged:
        exc_ind = (extns - n_start) * (n_start + extns + 1) // 2

        return psi[exc_ind + num_sig]  # returns the coefficient of interest

    return 0  # to account for other unphysical cases


@jit(nopython=True, cache=True)
def moments(pi1, pi2, sig1, sig2, lamb1, lamb2, phi, n_start, n_end):
    """
    Parameters
    ----------
    pi1: nonnegative integer (the number that appears as (ap)^{pi1} in the moments)
    pi2: nonnegative integer (the number that appears as (ap^dag)^{pi2} in the moments)
    sig1: nonnegative integer (the number that appears as (as)^{sig1} in the moments)
    sig2: nonnegative integer (the number that appears as (as^dag)^{sig2} in the moments)
    lamb1: nonnegative integer (the number that appears as (ai)^{lamb1} in the moments)
    lamb2: nonnegative integer (the number that appears as (ai^dag)^{lamb2} in the
                                moments)
    phi: 1d array with real numbers (the state of the system)
    n_start: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i};
     see Eq. (7b) in the manuscript)

    n_end: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i} )

    Returns
    -------
    moment: real number, the expectation value of the following quantity,
    <(ap^dag)^{pi2}*(ap)^{pi1}*(as^dag)^{sig2}*(as)^{sig1}
                           *(ai^dag)^{lamb2}*(ai)^{lamb1} >

    (Warning!: note that pi1 appears before pi2 in the input of the
     function argument, likewise for sig1 and sig2, lamb1 and lamb2,
     so be careful with the ordering of the input arguments for the function)
    """
    diff_pi = pi2 - pi1
    diff_sig = sig2 - sig1
    diff_lamb = lamb2 - lamb1

    if diff_sig != diff_lamb:  ## selection rules
        return 0

    moment = 0
    # See notes Moments.pdf in the folder for details about this formula
    for n_val in range(n_start, n_end + 1, 1):
        for k_val in range(0, n_val + 1, 1):
            if pi1 > n_val - k_val or sig1 > k_val or lamb1 > k_val:
                pass
            else:
                moment = moment + coef(
                    n_val - k_val, k_val, k_val, phi, n_start, n_end
                ) * coef(
                    n_val - k_val + diff_pi,
                    k_val + diff_sig,
                    k_val + diff_sig,
                    phi,
                    n_start,
                    n_end,
                ) * np.sqrt(
                    perm(n_val - k_val + diff_pi, pi2)
                ) * np.sqrt(
                    perm(n_val - k_val, pi1)
                ) * np.sqrt(
                    perm(k_val + diff_sig, sig2)
                ) * np.sqrt(
                    perm(k_val, sig1)
                ) * np.sqrt(
                    perm(k_val + diff_lamb, lamb2)
                ) * np.sqrt(
                    perm(k_val, lamb1)
                )
    return moment


@jit(nopython=True, cache=True)
def sig_mat_purity(phi, n_start, n_end):
    """
    Parameters
    ----------
    phi : 1 dimensional real array, the state vector of the system
    n_start: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i};
     see Eq. (7b) in the manuscript)

    n_end: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i} )

    Returns
    -------
    sig_mat_purity : real number smaller than or equal to 1,
    returns the purity of signal's density matrix after tracing out pump and idler

    """
    # rho_{s} is a diagonal matrix with diagonal elements (Eq. 47 in the manuscript)
    # <sig_num|rho_{s}|sig_num>=sum_{0}^{infty} |beta_{exc-sig_num,sig_num}|^{2}:

    smat_diag = np.zeros(n_end + 1, dtype=np.float64)
    for sig_num in range(0, n_end + 1):
        for exc in range(sig_num, sig_num + n_end + 1):
            smat_diag[sig_num] = (
                smat_diag[sig_num]
                + (coef(exc - sig_num, sig_num, sig_num, phi, n_start, n_end)) ** 2
            )

    smat_ans = np.dot(smat_diag, smat_diag)  # purity
    return smat_ans


@jit(nopython=True, cache=True)
def pump_mat_purity(psi, n_start, n_end):
    """
    Parameters
    ----------
    psi : a 1D array of real numbers, the state of the system.
    n_start: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i};
     see Eq. (7b) in the manuscript)

    n_end: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i} )

    Returns
    -------
    pump_mat_purity : returns a real number smaller than or equal to 1,
                 (purity of the pump matrix)

    """

    # A matrix is constructed with beta values, and this matrix is used for
    # the construction of rho_p. (see Eq. 46 in the manuscript)
    # Matrix, beta_pmat(num_pump,num_sig) = \beta_{num_pump,num_sig} in Eq. 46
    beta_pmat = np.zeros((n_end + 1, n_end + 1), dtype=np.float64)

    for num_pump in range(0, n_end + 1, 1):
        temp1 = max(0, n_start - num_pump)  # at other values coef is zero
        temp2 = n_end - num_pump  # at other values coef is zero
        for num_sig in range(temp1, temp2 + 1, 1):
            beta_pmat[num_pump, num_sig] = coef(
                num_pump, num_sig, num_sig, psi, n_start, n_end
            )

    # Construction of density matrix of rho_{p} (in Eq. 46)
    # everything is real so no conjugates are required:
    rho_p = beta_pmat @ np.transpose(beta_pmat)

    # Fun fact: the purity of rho_p obtained from beta_pmat @ np.transpose(beta_pmat)
    # and np.transpose(beta_pmat) @ beta_pmat are both equal, so if pump photon number
    # is along row or column does not make any difference to the computation of purity
    # This is because Tr(B @ B.T @ B @ B.T) = Tr(B.T @ B @ B.T @ B)

    # Purity of pump state:
    # We use a trick to speedup computing diagonal elements of rho squared here:
    # (AB)_{nn} = Sum_{p} (A.T)_{pn} B_{pn} and rho^{T}=rho here since rho is real
    pump_purity = np.sum(
        np.sum(rho_p * rho_p, 0)
    )  # same as np.trace(np.dot(pmat,pmat))

    return pump_purity


@jit(nopython=True, cache=True)
def pump_marg_prob(phi, num_pump, n_start, n_end):
    """
    Parameters
    ----------
    phi: a 1D array of real numbers, the state of the system.
    num_pump: nonnegative integer, the number of pump photons
    n_start: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i};
     see Eq. (7b) in the manuscript)

    n_end: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i} )

    Returns
    -------
    pump_prob : returns a positive real number smaller than or equal to one.
                (marginal probability of pump mode to have num_pump photons)

    """

    pump_prob = 0
    for num_signal in range(0, n_end + 1):
        num_idler = num_signal
        pump_prob = (
            pump_prob
            + (coef(num_pump, num_signal, num_idler, phi, n_start, n_end)) ** 2
        )

    return pump_prob


@jit(nopython=True, cache=True)
def signal_marg_prob(phi, num_signal, n_start, n_end):
    """
    Parameters
    ----------
    phi : a 1D array of real numbers, the state of the system.
    num_signal: nonnegative integer, the number of signal photons
    n_start: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i};
     see Eq. (7b) in the manuscript)

    n_end: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i} )

    Returns
    -------
    signal_prob : returns a real number
                (marginal probability of signal mode to have num_signal photons)

    """
    num_idler = num_signal
    signal_prob = 0
    for num_pump in range(0, n_end + 1):
        signal_prob = (
            signal_prob
            + (coef(num_pump, num_signal, num_idler, phi, n_start, n_end)) ** 2
        )

    return signal_prob


@jit(nopython=True, cache=True)
def witness_fourth_order(phi, n_start, n_end):
    """
    Parameters
    ----------
    phi : a 1D array of real numbers, the state of the system.

    n_start: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i};
     see Eq. (7b) in the manuscript)

    n_end: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i} )

    Returns
    -------
    real number (the value of fourth order witness)

    """

    # Refer to Eq. (E2) in Appendix E of the manuscript
    # We obtain the matrix elements of the witness below:
    a11 = 0.5 * (
        moments(1, 1, 1, 1, 0, 0, phi, n_start, n_end)
        + moments(1, 1, 0, 0, 1, 1, phi, n_start, n_end)
    )
    a12 = 0.5 * (
        moments(0, 2, 1, 1, 0, 0, phi, n_start, n_end)
        + moments(0, 2, 0, 0, 1, 1, phi, n_start, n_end)
    )
    a13 = moments(0, 2, 1, 0, 1, 0, phi, n_start, n_end)

    a22 = a11 + 0.5 * (
        moments(0, 0, 1, 1, 0, 0, phi, n_start, n_end)
        + moments(0, 0, 0, 0, 1, 1, phi, n_start, n_end)
    )
    a23 = moments(1, 1, 1, 0, 1, 0, phi, n_start, n_end) + moments(
        0, 0, 1, 0, 1, 0, phi, n_start, n_end
    )

    a33 = a22 + moments(1, 1, 0, 0, 0, 0, phi, n_start, n_end) + 1

    wit_mat = np.array([[a11, a12, a13], [0, a22, a23], [0, 0, a33]])

    # Using Hermiticity, we get the lower block elements of the matrix
    # and we have only real values:
    wit_mat = wit_mat + np.transpose(wit_mat) - np.diag(np.diag(wit_mat))

    return np.linalg.det(wit_mat)


@jit(nopython=True, cache=True)
def witness_sixth_order(phi, n_start, n_end):
    """
    Parameters
    ----------
    phi : a 1D array of real numbers, the state of the system.

    n_start: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i};
     see Eq. (7b) in the manuscript)

    n_end: integer
    (this number accounts for truncation of the Hilbert space as it appears in
     psi(t)=sum_{n=n_start}^{n_end} beta{n-k,k}|n-k>_{p} |k>_{s} |k>_{i} )

    Returns
    -------
    real number (the value of sixth order witness)

    """
    # Refer to Eq. (E3) in Appendix E of the manuscript
    # We obtain the matrix elements of the witness below:
    a11 = 1
    a12 = moments(1, 0, 0, 1, 0, 1, phi, n_start, n_end)
    a13 = moments(0, 1, 1, 0, 1, 0, phi, n_start, n_end)

    a22 = (
        0.25 * moments(1, 1, 0, 0, 2, 2, phi, n_start, n_end)
        + 0.25 * moments(1, 1, 2, 2, 0, 0, phi, n_start, n_end)
        + moments(1, 1, 1, 1, 1, 1, phi, n_start, n_end)
    )
    a23 = 1.5 * moments(0, 2, 2, 0, 2, 0, phi, n_start, n_end)

    a33 = (
        0.25
        * (
            moments(1, 1, 0, 0, 2, 2, phi, n_start, n_end)
            + 4 * moments(1, 1, 0, 0, 1, 1, phi, n_start, n_end)
            + 2 * moments(1, 1, 0, 0, 0, 0, phi, n_start, n_end)
            + moments(0, 0, 0, 0, 2, 2, phi, n_start, n_end)
            + 4 * moments(0, 0, 0, 0, 1, 1, phi, n_start, n_end)
            + 2
        )
        + 0.25
        * (
            moments(1, 1, 2, 2, 0, 0, phi, n_start, n_end)
            + 4 * moments(1, 1, 1, 1, 0, 0, phi, n_start, n_end)
            + 2 * moments(1, 1, 0, 0, 0, 0, phi, n_start, n_end)
            + moments(0, 0, 2, 2, 0, 0, phi, n_start, n_end)
            + 4 * moments(0, 0, 1, 1, 0, 0, phi, n_start, n_end)
            + 2
        )
        + (
            moments(1, 1, 1, 1, 1, 1, phi, n_start, n_end)
            + moments(1, 1, 1, 1, 0, 0, phi, n_start, n_end)
            + moments(1, 1, 0, 0, 1, 1, phi, n_start, n_end)
            + moments(1, 1, 0, 0, 0, 0, phi, n_start, n_end)
            + moments(0, 0, 1, 1, 1, 1, phi, n_start, n_end)
            + moments(0, 0, 1, 1, 0, 0, phi, n_start, n_end)
            + moments(0, 0, 0, 0, 1, 1, phi, n_start, n_end)
            + 1
        )
    )

    wit_mat = np.array([[a11, a12, a13], [0, a22, a23], [0, 0, a33]])

    # Using Hermiticity, we get the lower block elements of the matrix
    # and we have only real values:
    wit_mat = wit_mat + np.transpose(wit_mat) - np.diag(np.diag(wit_mat))

    return np.linalg.det(wit_mat)

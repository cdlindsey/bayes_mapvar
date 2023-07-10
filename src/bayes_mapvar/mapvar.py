''' Utilities for MAP and posterior variance estimation'''

import numpy as np
import pandas as pd
import scipy.optimize

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_mapvar.map_utils \
    import par_dict_from_vec,par_vec_from_dict, traverse_dist
from bayes_mapvar.var_utils \
    import get_hessian_delta_variance, get_bandwidths


tfd = tfp.distributions
tfb = tfp.bijectors


def mapvar(dist_dict,
            observed_data,
            observed_varnames=None,
            constrained_fcns=None,
            skip_var=True):
    """ Estimate posterior modes and posterior variances

        Parameters:
        -------
        dist_dict: dict
            dictonary of distributions representing
            unconstrained parameters (prior) and observed data
             likelihood, jointly the posterior.

            Constrained parametrers may be represented in this
            dictionary as callables that return a Deterministic
            distribution (so dist_dict could be used with
            tfp.JointDistributionNamed).

        observed_data: dict
            dictionary of observed data, used in calculating the
            posterior density

        observed_varnames: list
            list of observed varaibles in dist_dict

        constrained_fcns: dictionary (optional)
            dictionary of functions mapping the unconstrained
            parameters to the constrained and transformed parameters

        calc_log_prob: boolean, default value = False
            skip posterior variance estimation?

        Returns:
        -------

        unconstrained_par_map: dictionary
            dictionary of unconstrained parameter value
            posterior mode

        constrained_par_map: dictionary
            dictionary of constrained parameters from the
            unconstrained posterior mode

        scipyopt: SciPy optimization object
            results of MAP estimation.

        hessian: dataframe
            Hessian matrix of log posterior density.
            Dataframe is indexed by unconstrained parameter
            names and indices to flattened parameter vectors.

        delta: dataframe
            Delta matrix for constrained parameter variance
            calculation.
            Dataframe is indexed by constrained and
            unconstrained parameter names and indices to
            flattened parameter vectors.

        constrained_var: dataframe
            Constrained variance matrix.
            Dataframe is indexed by constrained parameter names and
            indices to flattened parameter vectors.
    """
    _, init_unconstrained_par_dict, _, _ = \
        traverse_dist(dist_dict,
                        observed_data,
                        observed_varnames,
                        generate="unconstrained",
                        constrained_fcns=constrained_fcns,
                        calc_log_prob = False,
                        unconstrained_par_dict=None,
                        unconstrained_generate="zero")
    def loss(par_vec):
        unconstrained_par_dict = \
            par_dict_from_vec(par_vec,
                init_unconstrained_par_dict)
        log_prob, _, _, _ = \
            traverse_dist(dist_dict,
                            observed_data,
                            observed_varnames,
                            generate="constrained",
                            constrained_fcns=constrained_fcns,
                            calc_log_prob = True,
                            unconstrained_par_dict=
                                unconstrained_par_dict)
        return -log_prob

    def np_loss_and_gradient(par_vec):
        loss_gradient = tfp.math.value_and_gradient(loss, tf.convert_to_tensor(par_vec,tf.float64))
        return np.array(tf.cast(loss_gradient[0], tf.float64).numpy()), \
            tf.cast(loss_gradient[1], tf.float64).numpy()

    scipyopt = scipy.optimize.minimize(
                    np_loss_and_gradient,
                    jac=True,
                    x0=par_vec_from_dict(
                        init_unconstrained_par_dict),
                    method='L-BFGS-B')

    unconstrained_par_map = par_dict_from_vec(
        scipyopt.x,init_unconstrained_par_dict)
    _, _, constrained_par_map, _ = \
        traverse_dist(dist_dict,
                        observed_data,
                        observed_varnames,
                        generate="constrained",
                        constrained_fcns=constrained_fcns,
                        calc_log_prob = False,
                        unconstrained_par_dict=
                        unconstrained_par_map)
    def constrained_vec(par_vec):
        unconstrained_par_dict = par_dict_from_vec(
            par_vec,init_unconstrained_par_dict)
        _, _, constrained_par_dict, _ = \
            traverse_dist(dist_dict,
                            observed_data,
                            observed_varnames,
                            generate="constrained",
                            constrained_fcns=constrained_fcns,
                            calc_log_prob = False,
                            unconstrained_par_dict=
                                unconstrained_par_dict)
        return par_vec_from_dict(constrained_par_dict)
    hessian = None
    delta = None
    variance = None
    if not skip_var:
        constrained_par_size = \
            {key: tf.size(value) for key, value in
                constrained_par_map.items()}
        unconstrained_par_size = \
            {key: tf.size(value) for key, value in
                unconstrained_par_map.items()}
        bandwidths = get_bandwidths(
            par_vec_from_dict(unconstrained_par_map))
        hessian, delta, variance = \
            get_hessian_delta_variance(
                par_vec_from_dict(unconstrained_par_map).numpy(),
                loss,
                constrained_vec,
                bandwidths,
                unconstrained_par_size,
                constrained_par_size)
                
    return unconstrained_par_map, \
            constrained_par_map, \
            scipyopt, \
            hessian, \
            delta, \
            variance

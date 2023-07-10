''' Utilities for Maximum A Posteriori (MAP) estimation'''
from inspect import signature
import numpy as np
import pandas as pd
import scipy.optimize

import tensorflow as tf
import tensorflow_probability as tfp

from bayes_mapvar.exceptions import MapVarException

tfd = tfp.distributions
tfb = tfp.bijectors

def par_vec_from_dict(par_dict):
    """ Get parameter vector from parameter dictionary

        Parameters:
        -------
        par_dict: dictionary
            dictionary of parameter values

        Returns:
        -------
        Parameter vector
    """
    keylist = list(par_dict.keys())
    init_vec = tf.zeros(0,dtype=tf.float64)
    for key in keylist:
        init_vec = tf.concat([init_vec,tf.reshape(par_dict[key],-1)], 0)
    return init_vec

def par_dict_from_vec(par_vec, par_dict_ref):
    """ Get parameter dictionary from parameter vector

        Parameters:
        -------
        par_vec: 1D tensor
            vector of parameter values

       par_dict_ref: dictionary
            dictionary of parameter values, used only for
            shape/size information

        Returns:
        -------
        Parameter dictionary
    """
    pardict = {}
    parindex = 0
    for key in par_dict_ref.keys():
        keysize = tf.size(par_dict_ref[key])
        keyshape = par_dict_ref[key].shape
        keypar = par_vec[parindex:(parindex + keysize)]
        parindex = parindex + keysize
        pardict[key] = tf.reshape(keypar, keyshape)
    return pardict

def traverse_dist(dist_dict, observed_data, observed_varnames,
                  generate, constrained_fcns=None, calc_log_prob = True,
                  unconstrained_par_dict=None, unconstrained_generate="zero"):
    """ Calculate constrained parameters and evaluate posterior density

        Parameters:
        -------
        dist_dict: dict
            dictonary of distributions representing unconstrained parameters
            (prior) and observed data likliehood, jointly the posterior.

            Constrained parametrers may be represented in this dictionary as
            callables that return a Deterministic distribution (so dist_dict
            could be used with JointDistributionNamed).

        observed_data: dict
            dictionary of observed data, used in calculating the posterior
            density

        observed_varnames: list
            list of observed varaibles in dist_dict

        generate: string
            Takes one of three values, indicating what should be generated
            during the traversal.

            "unconstrained" : generate a dictionary of unconstrained
                              parameter values, these may be used as
                              initial values in optimization

            "constrained" : generate a dictionary of constrained parameter
                            values from the input unconstrained_par_dict
                            unconstrained parameter values

            "post pred" : generate a dictionary containing the constrained
                          parameter values and new samples of the observed
                          variables based on the input unconstrained_par_dict
                          unconstrained parameter values.
                          These are posterior predictive draws of the
                          observed variables.

        constrained_fcns: dictionary (optional)
            dictionary of functions mapping the unconstrained parameters
            to the constrained and transformed parameters

        calc_log_prob: boolean, default value = True
            calculate the posterior density?

        unconstrained_par_dict: dictionary (optional)
            dictionary of unconstrained parameter values

        unconstrained_generate: string, default value = "zero"
            If generate="unconstrained", indicates what method to use for
            obtaining unconstrained parameter values.
            "zero": all unconstrained parameter values are zero
            "sample mean": unconstrained parameters are obtained by taking
                           the mean of 100 samples of their distribution.

        Returns:
        -------

        log_prob: scalar
            log probability density of posterior if calculated, otherwise 0

        unconstrained_par_dict: dictionary
            dictionary of unconstrained parameter values

        constrained_par_dict: dictionary
            dictionary of constrained parameter values

        post_pred_dict: dictionary
            dictionary of posterior predictive samples of observed data

    """
    log_prob_res = tf.zeros(1,dtype=tf.float64)
    if generate == "unconstrained" and unconstrained_par_dict is not None:
        raise MapVarException('cannot specify unconstrained_generate and unconstrained_par_dict simultaneously')

    if unconstrained_generate not in ["zero", "sample_mean"]:
        raise MapVarException('invalid value for generate argument')

    if generate not in ["unconstrained","constrained","post_pred"]:
        raise MapVarException('invalid value for generate argument')

    constrained_par_dict = {}
    post_pred_dict = {}
    if generate == "unconstrained":
        unconstrained_par_dict = {}

    if constrained_fcns is None:
        constrained_fcns = {}

    vars_pars_to_do = list(dist_dict.keys()) + list(constrained_fcns.keys())

    samp_dict = {}
    while len(vars_pars_to_do) > 0:
        var_par = vars_pars_to_do[0]
        arg_dict = {}
        arg_list = []
        addback2vars_pars_to_do = False
        samp_caller = None
        if var_par in constrained_fcns:
            samp_caller = constrained_fcns[var_par]
            arg_list = list(signature(samp_caller).parameters.keys())
        elif callable(dist_dict[var_par]):
            samp_caller = dist_dict[var_par]
            arg_list = list(signature(samp_caller).parameters.keys())
        elif var_par not in observed_varnames:
            if generate == "unconstrained":
                if unconstrained_generate == "zero":
                    samp_dict[var_par] = tf.zeros(
                        dist_dict[var_par].batch_shape +
                        dist_dict[var_par].event_shape,dtype=tf.float64)
                else:
                    samp_dict[var_par] = tf.reduce_mean(dist_dist[var_par].sample(100),axis=0)
                unconstrained_par_dict[var_par] = samp_dict[var_par]
            else:
                samp_dict[var_par] = unconstrained_par_dict[var_par]
            if calc_log_prob:
                log_prob_res += tf.reduce_sum(
                            dist_dict[var_par].log_prob(samp_dict[var_par]))
        else:
            if generate == "pred_post":
                samp_dict[var_par] = dist_dict[var_par].sample()
                post_pred_dict[var_par] = samp_dict[var_par]
            else:
                samp_dict[var_par] = observed_data[var_par]
            if calc_log_prob:
                log_prob_res += tf.reduce_sum(
                            dist_dict[var_par].log_prob(samp_dict[var_par]))
        for arg in arg_list:
            if arg not in samp_dict:
                addback2vars_pars_to_do = True
                vars_pars_to_do_tmp = [arg]
                vars_pars_to_do_tmp.extend(vars_pars_to_do)
                vars_pars_to_do =  vars_pars_to_do_tmp
            else:
                arg_dict[arg] = samp_dict[arg]
        if not addback2vars_pars_to_do and samp_caller is not None:
            if calc_log_prob or not (var_par in observed_varnames
                                     and generate != "post pred"):
                samp_res_caller = samp_caller(**arg_dict)
            elif var_par in observed_varnames and generate != "post pred":
                samp_dict[var_par] = observed_data[var_par]
            if var_par in constrained_fcns:
                samp_dict[var_par] = samp_res_caller
                if generate == "constrained" or generate == "post pred":
                    constrained_par_dict[var_par] = samp_dict[var_par]
            elif var_par in observed_varnames and generate == "post pred":
                samp_dict[var_par] = samp_res_caller.sample()
                post_pred_dict[var_par] = samp_dict[var_par]
            elif var_par not in observed_varnames:
                if samp_res_caller.name == 'Deterministic':
                    samp_dict[var_par] = samp_res_caller.loc
                    if generate == "constrained" or generate == "post pred":
                        constrained_par_dict[var_par] = samp_dict[var_par]
                else:
                    if generate == "unconstrained":
                        if unconstrained_generate == "zero":
                            samp_dict[var_par] = tf.zeros(
                            samp_res_caller.batch_shape +
                            samp_res_caller.event_shape,dtype=tf.float64)
                        else:
                            samp_dict[var_par] = tf.reduce_mean(
                                samp_res_caller.sample(100),axis=0)
                        unconstrained_par_dict[var_par] = samp_dict[var_par]
                    else:
                        samp_dict[var_par] = unconstrained_par_dict[var_par]
            else:
                samp_dict[var_par] = observed_data[var_par]
            if calc_log_prob and var_par not in constrained_fcns:
                log_prob_res += tf.reduce_sum(samp_res_caller.log_prob(samp_dict[var_par]))
        if var_par in samp_dict:
            vars_pars_to_do = list(set(vars_pars_to_do) - set([var_par]))
    return log_prob_res, unconstrained_par_dict, constrained_par_dict, post_pred_dict

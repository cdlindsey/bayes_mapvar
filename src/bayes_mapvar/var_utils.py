''' Utilities for posterior variance estimation'''
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

def get_bandwidths(unconstrained_par_vec):
    """ Get bandwidths for calculating numeric derivatives w.r.t. the parameters
    """
    abspars = abs(unconstrained_par_vec)
    epsdouble = np.finfo(float).eps
    epsdouble = epsdouble**(1 / 3)
    scale = epsdouble * (abspars + epsdouble)
    scaleparmstable = scale + abspars
    return scaleparmstable - abspars

def get_hessian_delta_variance(unconstrained_par_vec, loss,
        constrained_par_vec_fcn, bandwidths,
        unconstrained_par_size, constrained_par_size):
    """ Get Hessian for posterior and Delta matrix for constrained
        variance calculation

        Parameters:
        -------
        unconstrained_par_vec: vector
            unconstrained parameter values

        loss: function
            negative of log posterior density function

        constrained_par_vec_fcn: function
            returns constrained parameters in a vector given
            unconstrained

        bandwidths: vector
            bandwidths to use in numerical derivatives for
            Hessian and Delta calculation

        unconstrained_par_size: dict
            dictionary of sizes of unconstrained parameters,
            used in labeling hessian and delta matrices.

        constrained_par_size: dict
            dictionary of sizes of constrained parameters,
            used in labeling unconstrained variance matrix.

        Returns:
        -------
        hessian: dataframe
            Hessian matrix of log posterior density.
            Dataframe is indexed by unconstrained parameter
            names and indices to flattened parameter vectors.

        delta: dataframe
            Delta matrix for constrained parameter variance
            calculation.
            Dataframe is indexed by constrained and unconstrained
            parameter names and indices to flattened parameter
            vectors.

        constrained_var: dataframe
            Constrained variance matrix.
            Dataframe is indexed by constrained parameter names and
            indices to flattened parameter vectors.
    """
    npar = len(unconstrained_par_vec)
    hessian = np.zeros((npar,npar))
    for iter_par in np.arange(0, npar):
        parplus = unconstrained_par_vec.copy()
        parplus[iter_par] = parplus[iter_par] \
            + bandwidths[iter_par]
        parminus = unconstrained_par_vec.copy()
        parminus[iter_par] = parminus[iter_par] \
            - bandwidths[iter_par]
        gradplus = tfp.math.value_and_gradient(loss,
            tf.convert_to_tensor(parplus,tf.float64))[1]
        gradminus = tfp.math.value_and_gradient(loss,
            tf.convert_to_tensor(parminus,tf.float64))[1]
        hessian[:, iter_par] = (gradplus - gradminus) / (
                                2 * bandwidths[iter_par])
        consplus = constrained_par_vec_fcn(parplus)
        consminus = constrained_par_vec_fcn(parminus)
        if iter_par == 0:
            delta = np.zeros((len(consplus), npar))
        delta[:, iter_par] = (consplus - consminus) / (
                            2 * bandwidths[iter_par])
    hessian = (hessian+np.transpose(hessian))/2
    constrained_var = np.matmul(
                        np.matmul(delta,np.linalg.inv(hessian)),
                            np.transpose(delta))
    constrained_labels = []
    for constrained_par in constrained_par_size:
        size_constrained_par = \
            constrained_par_size[constrained_par]
        if size_constrained_par > 1:
            for i in range(0,size_constrained_par):
                constrained_labels.append(
                    constrained_par +"_" + str(i))
        else:
            constrained_labels.append(constrained_par)
    constrained_var = pd.DataFrame(
                        constrained_var,index=constrained_labels,
                            columns=constrained_labels)
    unconstrained_labels = []
    for unconstrained_par in unconstrained_par_size:
        size_unconstrained_par = \
            unconstrained_par_size[unconstrained_par]
        if size_unconstrained_par > 1:
            for i in range(0,size_unconstrained_par):
                unconstrained_labels.append(
                    unconstrained_par +"_" + str(i))
        else:
            unconstrained_labels.append(unconstrained_par)
    hessian = pd.DataFrame(hessian,
                index=unconstrained_labels,
                    columns=unconstrained_labels)
    delta = pd.DataFrame(delta,
                index=constrained_labels,
                    columns=unconstrained_labels)
    return hessian, delta, constrained_var

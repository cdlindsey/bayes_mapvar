''' Unit tests for MAP and posterior variance estimation '''
import os
from unittest import TestCase
import pytest
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

from tests.data.load_data_csv import load_data_csv
from tests.test_utils import reldif
from bayes_mapvar.mapvar import mapvar

tfd = tfp.distributions
tfb = tfp.bijectors

class TestMapVar(TestCase):
    ''' Unit tests for MapVar estimation '''
    
    @classmethod
    def setUpClass(self):  # pylint: disable=bad-classmethod-argument
        self.samp_data = load_data_csv("sim_ex.csv")


    @pytest.mark.eager
    def test_mapvar_sim(self):
    
        dist_dict = {}
        dist_dict['beta'] = tfd.Normal(tf.ones(1,dtype=tf.float64),1)
        dist_dict['unconstrained_alpha'] = tfd.TransformedDistribution(
            tfd.Chi2(4*tf.ones(1,dtype=tf.float64)),tfb.Log())
        dist_dict['alpha'] = lambda unconstrained_alpha: \
            tfd.Deterministic(loc=tfb.Log().inverse(unconstrained_alpha))
        dist_dict['y'] = lambda alpha, beta: \
            tfd.Normal(loc = alpha + beta*self.samp_data['x'],scale=tf.ones(1,dtype=tf.float64))
    
        m0 = mapvar(dist_dict,
            self.samp_data,
            observed_varnames=['y'], skip_var=False)

        assert reldif(m0[0]['unconstrained_alpha'].numpy()[0],
            0.99147004) < 1e-4, "map and posterior variance estimation failed"

        assert reldif(m0[0]['beta'].numpy()[0],1.51747775) < 1e-4, \
            "map and posterior variance estimation failed"

        assert reldif(m0[5].loc['alpha','alpha'], \
            0.009973651034083814) < 1e-4, \
            "map and posterior variance estimation failed"
    
        assert reldif(m0[4].loc['alpha','unconstrained_alpha'], \
            2.695193593504064) < 1e-4, \
            "map and posterior variance estimation failed"

        dist_dict['unconstrained_beta'] = dist_dict['beta']
        dist_dict['beta'] = lambda unconstrained_beta: \
            tfd.Deterministic(loc=unconstrained_beta)

        m1 = mapvar(dist_dict, self.samp_data,
            observed_varnames=['y'], skip_var=False)

        assert reldif(m1[0]['unconstrained_alpha'].numpy()[0], \
            0.99147004) < 1e-4, "map and posterior variance estimation failed"

        assert reldif(m1[0]['unconstrained_beta'].numpy()[0], \
            1.51747775) < 1e-4, "map and posterior variance estimation failed"

        assert reldif(m1[5].loc['alpha','alpha'], \
            0.009973651034083814) < 1e-4, \
            "map and posterior variance estimation failed"

        assert reldif(m1[5].loc['beta','beta'], \
            0.009761802606078345) < 1e-4, \
            "map and posterior variance estimation failed"

        assert reldif(m1[5].loc['beta','alpha'], \
            0.00010401063160407793) < 1e-4, \
            "map and posterior variance estimation failed"


        assert reldif(m1[4].loc['alpha','unconstrained_alpha'], \
            2.695193593504064) < 1e-4, \
            "map and posterior variance estimation failed"


        constraints = {}
        constraints['beta'] = lambda unconstrained_beta: unconstrained_beta
        constraints['alpha'] = lambda unconstrained_alpha: \
            tfb.Log().inverse(unconstrained_alpha)

        del dist_dict['alpha']
        del dist_dict['beta']

        m2 = mapvar(dist_dict, self.samp_data,
            observed_varnames=['y'],
            constrained_fcns=constraints, skip_var=False)

        assert reldif(m2[0]['unconstrained_alpha'].numpy()[0], \
            0.99147004)  < 1e-4, \
            "map and posterior variance estimation failed"

        assert reldif(m2[0]['unconstrained_beta'].numpy()[0], \
            1.51747775) < 1e-4, "map and posterior variance estimation failed"

        assert reldif(m2[5].loc['alpha','alpha'], \
            0.009973651034083814)  < 1e-4, \
            "map and posterior variance estimation failed"

        assert reldif(m2[5].loc['beta','beta'], \
            0.009761802606078345) < 1e-4, \
            "map and posterior variance estimation failed"

        assert reldif(m2[5].loc['beta','alpha'], \
            0.00010401063160407793) < 1e-4, \
            "map and posterior variance estimation failed"

        assert reldif(m2[4].loc['alpha','unconstrained_alpha'], \
            2.695193593504064) < 1e-4, \
            "map and posterior variance estimation failed"

        assert reldif(m2[3].loc['unconstrained_beta', \
            'unconstrained_alpha'],-2.879597369813636) < 1e-4, \
            "map and posterior variance estimation failed"

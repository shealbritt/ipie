# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Joonho Lee
#          Fionn Malone <fionn.malone@gmail.com>
#

import numpy
import pytest

from ipie.estimators.greens_function import greens_function_single_det
from ipie.estimators.greens_function_single_det import greens_function_single_det_ghf
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.misc import dotdict
from ipie.utils.mpi import MPIHandler
from ipie.utils.testing import build_test_case_handlers
from ipie.walkers.ghf_walkers import GHFWalkers


@pytest.mark.unit
def test_overlap_greens_function():
    nelec = (7, 5)
    nwalkers = 10
    nsteps = 0
    nmo = 10
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        complex_trial=False,
        options=qmc,
        seed=7,
        reortho=False,
        trial_type="single_det",
    )
    uhf_trial = batched_data.trial
    uhf_walkers = batched_data.walkers

    # Define GHF wavefunctions from UHF.
    ghf_psi0 = numpy.zeros((2 * nmo, numpy.sum(nelec)), dtype=uhf_trial.psi0a.dtype)
    ghf_psi0[:nmo, : nelec[0]] = uhf_trial.psi0a.copy()
    ghf_psi0[nmo:, nelec[0] :] = uhf_trial.psi0b.copy()

    ghf_phi = numpy.zeros((2 * nmo, numpy.sum(nelec)), dtype=uhf_walkers.phia.dtype)
    ghf_phi[:nmo, : nelec[0]] = uhf_walkers.phia[0].copy()
    ghf_phi[nmo:, nelec[0] :] = uhf_walkers.phib[0].copy()

    ghf_trial = SingleDetGHF(ghf_psi0, nelec, nmo)
    ghf_walkers = GHFWalkers(ghf_phi, nelec[0], nelec[1], nmo, nwalkers)
    ghf_walkers.build(ghf_trial)

    ovlp = greens_function_single_det(uhf_walkers, uhf_trial, build_full=True)
    ovlp_ghf = greens_function_single_det_ghf(ghf_walkers, ghf_trial)
    numpy.testing.assert_allclose(ovlp, ovlp_ghf, atol=1e-10)

    uhf_walkers.ovlp = uhf_trial.calc_overlap(uhf_walkers)
    ghf_walkers.ovlp = ghf_trial.calc_overlap(ghf_walkers)
    numpy.testing.assert_allclose(uhf_walkers.ovlp, ghf_walkers.ovlp, atol=1e-10)

    numpy.testing.assert_allclose(uhf_trial.psi0a, ghf_trial.psi0[:nmo, : nelec[0]], atol=1e-10)
    numpy.testing.assert_allclose(uhf_trial.psi0b, ghf_trial.psi0[nmo:, nelec[0] :], atol=1e-10)
    numpy.testing.assert_allclose(
        uhf_walkers.phia, ghf_walkers.phi[:, :nmo, : nelec[0]], atol=1e-10
    )
    numpy.testing.assert_allclose(
        uhf_walkers.phib, ghf_walkers.phi[:, nmo:, nelec[0] :], atol=1e-10
    )
    numpy.testing.assert_allclose(uhf_walkers.Ga, ghf_walkers.G[:, :nmo, :nmo], atol=1e-10)
    numpy.testing.assert_allclose(uhf_walkers.Gb, ghf_walkers.G[:, nmo:, nmo:], atol=1e-10)
    numpy.testing.assert_allclose(uhf_walkers.Ga, ghf_walkers.Ga, atol=1e-10)
    numpy.testing.assert_allclose(uhf_walkers.Gb, ghf_walkers.Gb, atol=1e-10)


@pytest.mark.unit
def test_ghf_walkers_from_uhf_walkers():
    nelec = (7, 5)
    nwalkers = 10
    nsteps = 10
    nmo = 10
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        complex_trial=False,
        options=qmc,
        seed=7,
        reortho=False,
        trial_type="single_det",
    )
    uhf_trial = batched_data.trial
    uhf_walkers = batched_data.walkers

    ghf_trial = SingleDetGHF(uhf_trial)
    ghf_walkers = GHFWalkers(uhf_walkers)

    ovlp = greens_function_single_det(uhf_walkers, uhf_trial, build_full=True)
    ovlp_ghf = greens_function_single_det_ghf(ghf_walkers, ghf_trial)
    numpy.testing.assert_allclose(ovlp, ovlp_ghf, atol=1e-10)

    uhf_walkers.ovlp = uhf_trial.calc_overlap(uhf_walkers)
    ghf_walkers.ovlp = ghf_trial.calc_overlap(ghf_walkers)
    numpy.testing.assert_allclose(uhf_walkers.ovlp, ghf_walkers.ovlp, atol=1e-10)

    numpy.testing.assert_allclose(
        uhf_walkers.phia, ghf_walkers.phi[:, :nmo, : nelec[0]], atol=1e-10
    )
    numpy.testing.assert_allclose(
        uhf_walkers.phib, ghf_walkers.phi[:, nmo:, nelec[0] :], atol=1e-10
    )
    numpy.testing.assert_allclose(uhf_walkers.Ga, ghf_walkers.G[:, :nmo, :nmo], atol=1e-10)
    numpy.testing.assert_allclose(uhf_walkers.Gb, ghf_walkers.G[:, nmo:, nmo:], atol=1e-10)
    numpy.testing.assert_allclose(uhf_walkers.Ga, ghf_walkers.Ga, atol=1e-10)
    numpy.testing.assert_allclose(uhf_walkers.Gb, ghf_walkers.Gb, atol=1e-10)


@pytest.mark.unit
def test_reortho_batch():
    nelec = (7, 5)
    nwalkers = 10
    nsteps = 10
    nmo = 10
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    batched_data = build_test_case_handlers(
        nelec, nmo, num_dets=1, complex_trial=True, options=qmc, seed=7, reortho=False
    )
    uhf_walkers = batched_data.walkers
    ghf_walkers = GHFWalkers(uhf_walkers)

    detR = uhf_walkers.reortho()
    detR_ghf = ghf_walkers.reortho()

    numpy.testing.assert_allclose(detR, detR_ghf, atol=1e-10)


if __name__ == "__main__":
    test_overlap_greens_function()
    test_ghf_walkers_from_uhf_walkers()
    test_reortho_batch()

"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

import numpy as np
import pytest

from qrisp import QuantumFloat
from qrisp.operators import X, Z
from qrisp.qml import Encoding, expectation

# Coarse enough to keep the suite fast, tight enough that a wrong sign or a
# swapped entry cannot pass.
PRECISION = 0.02
TOLERANCE = 0.1


def state_prep(x):
    qf = QuantumFloat(2)
    Encoding.angle(n_features=2).apply(qf, x)
    return qf


@pytest.mark.parametrize(
    "x, observable, expected",
    [
        ([0.0, 0.0], Z(0), 1.0),
        ([np.pi, 0.0], Z(0), -1.0),
        ([np.pi, 0.0], Z(0) * Z(1), -1.0),
        ([0.0, 0.0], X(0), 0.0),
    ],
)
def test_expectation_of_a_single_observable_is_a_scalar(x, observable, expected):
    """
    A single observable is passed through to the operator, which yields one
    number rather than an array.
    """
    result = expectation(state_prep, observable, precision=PRECISION)(x)

    assert np.ndim(result) == 0
    assert np.isclose(result, expected, atol=TOLERANCE)


def test_expectation_of_a_sequence_keeps_the_entries_separate():
    """
    Each observable is reported on its own. The two qubits are in opposite
    states here, which a summed operator would report as a single 0.
    """
    result = expectation(state_prep, [Z(0), Z(1)], precision=PRECISION)([np.pi, 0.0])

    assert np.allclose(result, [-1.0, 1.0], atol=TOLERANCE)


def test_expectation_follows_the_order_of_the_observables():
    """
    The entries are aligned with the observables as given, so reversing the
    sequence reverses the result.
    """
    x = [np.pi, 0.0]

    forward = expectation(state_prep, [Z(0), Z(1)], precision=PRECISION)(x)
    reverse = expectation(state_prep, [Z(1), Z(0)], precision=PRECISION)(x)

    assert np.allclose(forward, reverse[::-1], atol=TOLERANCE)


def test_expectation_handles_non_commuting_observables():
    """
    Observables that cannot be measured in one circuit are estimated
    separately, so they may be combined freely.
    """
    result = expectation(state_prep, [Z(0), X(0)], precision=PRECISION)([0.0, 0.0])

    assert np.allclose(result, [1.0, 0.0], atol=TOLERANCE)


def test_expectation_accepts_a_tuple_of_observables():
    """Any sequence is accepted, not only a list."""
    result = expectation(state_prep, (Z(0), Z(1)), precision=PRECISION)([np.pi, 0.0])

    assert np.allclose(result, [-1.0, 1.0], atol=TOLERANCE)


@pytest.mark.parametrize("container", [list, tuple, np.array, iter], ids=["list", "tuple", "ndarray", "iterator"])
def test_expectation_accepts_any_sequence_of_observables(container):
    """
    Observables were identified by isinstance against list and tuple, so an
    ndarray or a generator fell through to the single operator branch and
    failed with an AttributeError naming a numpy internal.
    """
    observables = container([Z(0), Z(1)])

    def state_prep():
        return QuantumFloat(2)

    # The register stays in |00>, so both qubits are measured as +1.
    assert np.allclose(expectation(state_prep, observables)(), [1.0, 1.0])


def test_expectation_rejects_an_empty_sequence():
    """
    An empty sequence returned an empty array from a function that samples
    nothing, which reads as a successful measurement of no observables rather
    than as the mistake it is.
    """

    def state_prep():
        return QuantumFloat(2)

    with pytest.raises(ValueError, match="empty sequence"):
        expectation(state_prep, [])

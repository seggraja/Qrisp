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

import jax.numpy as jnp
import numpy as np
import pytest

from qrisp import QuantumFloat, measure
from qrisp.jasp import jaspify
from qrisp.qml import hardware_efficient, hardware_efficient_size


def statevector_array(qv):
    return np.array(qv.qs.statevector("array"), dtype=complex)


@pytest.mark.parametrize(
    "num_qubits, layers, rotations, expected",
    [(3, 2, ("y",), 9), (3, 2, ("y", "z"), 18), (4, 1, ("y",), 8), (2, 0, ("y",), 2)],
)
def test_hardware_efficient_size(num_qubits, layers, rotations, expected):
    """
    The ansatz uses one parameter per qubit, rotation axis and rotation layer,
    with one more rotation layer than entangling layers.
    """
    assert hardware_efficient_size(num_qubits, layers, rotations) == expected


def test_hardware_efficient_accepts_flat_and_shaped_parameters():
    """
    Parameters may be passed as a flat vector, as required by the optimizers,
    or shaped as (layers + 1, len(rotations), n).
    """
    w = np.linspace(0.1, 0.9, 9)

    qf_flat = QuantumFloat(3)
    hardware_efficient(qf_flat, w, layers=2)

    qf_shaped = QuantumFloat(3)
    hardware_efficient(qf_shaped, w.reshape(3, 1, 3), layers=2)

    assert np.allclose(statevector_array(qf_flat), statevector_array(qf_shaped))


def test_hardware_efficient_zero_parameters_is_identity():
    """
    With vanishing rotation angles only the entangling layers remain, which act
    trivially on |0>.
    """
    qf = QuantumFloat(3)
    hardware_efficient(qf, np.zeros(9), layers=2)

    expected = np.zeros(8, dtype=complex)
    expected[0] = 1.0

    assert np.allclose(statevector_array(qf), expected)


def test_hardware_efficient_single_qubit_matches_rotation_product():
    """
    On one qubit there is nothing to entangle, so the layers reduce to a single
    RY rotation by the sum of the parameters.
    """
    w = np.array([0.3, 0.7, 1.1])

    qf = QuantumFloat(1)
    hardware_efficient(qf, w, layers=2)

    total = w.sum()
    expected = np.array([np.cos(total / 2), np.sin(total / 2)])

    assert np.allclose(statevector_array(qf), expected)


def test_hardware_efficient_entangler_creates_entanglement():
    """
    A rotated first qubit followed by the CX chain produces a correlated state
    that does not factorize.
    """
    w = np.zeros(4)
    w[0] = np.pi / 2

    qf = QuantumFloat(2)
    hardware_efficient(qf, w, layers=1)

    # |00> and |11> are populated, the anticorrelated states are not.
    amplitudes = np.abs(statevector_array(qf))
    assert np.isclose(amplitudes[0], 1 / np.sqrt(2))
    assert np.isclose(amplitudes[3], 1 / np.sqrt(2))
    assert np.allclose(amplitudes[[1, 2]], 0.0)


@pytest.mark.parametrize("entangler", ["cx", "cz"])
def test_hardware_efficient_entanglers_differ(entangler):
    """Both entanglers are applied and produce distinct circuits."""
    w = np.linspace(0.1, 0.8, 8)

    qf = QuantumFloat(4)
    hardware_efficient(qf, w, layers=1, entangler=entangler)

    assert len(qf.qs.compile().qubits) == 4


def test_hardware_efficient_rejects_parameter_count_mismatch():
    """The parameter count is determined by the qubits, layers and rotations."""
    with pytest.raises(ValueError, match="Expected 9 parameters"):
        hardware_efficient(QuantumFloat(3), np.zeros(7), layers=2)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"rotations": ("w",)}, "Unknown rotation axis"),
        ({"entangler": "swap"}, "Unknown entangler"),
    ],
)
def test_hardware_efficient_rejects_unknown_components(kwargs, match):
    """Rotation axes and entanglers are restricted to the supported gates."""
    with pytest.raises(ValueError, match=match):
        hardware_efficient(QuantumFloat(2), np.zeros(4), layers=1, **kwargs)


def test_hardware_efficient_traces():
    """The ansatz is compatible with the Jasp tracing path."""

    @jaspify
    def main(w):
        qf = QuantumFloat(2)
        hardware_efficient(qf, w, layers=1)
        return measure(qf)

    w = jnp.zeros(4).at[0].set(jnp.pi)

    # RY(pi) on the first qubit followed by the CX chain flips both qubits.
    assert main(w) == 3

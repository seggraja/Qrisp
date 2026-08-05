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

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from qrisp import BasisPreparationWarning, QuantumFloat, measure
from qrisp.jasp import jaspify
from qrisp.qml import angle_encoding


def statevector_array(qv):
    return np.array(qv.qs.statevector("array"), dtype=complex)


@pytest.mark.parametrize(
    "features",
    [
        [0.0, 0.0],
        [0.0, np.pi],
        [np.pi / 2, np.pi / 3],
        [-0.5, 2.1],
    ],
)
def test_angle_encoding_ry_amplitudes(features):
    """
    RY angle encoding prepares the product state
    tensor_i (cos(x_i/2)|0> + sin(x_i/2)|1>).
    """
    qf = QuantumFloat(len(features))
    angle_encoding(qf, features)

    # In the statevector array the first feature corresponds to the most
    # significant bit, so the Kronecker product follows the feature order.
    expected = np.array([1.0], dtype=complex)
    for x in features:
        expected = np.kron(expected, np.array([np.cos(x / 2), np.sin(x / 2)]))

    assert np.allclose(statevector_array(qf), expected)


def test_angle_encoding_rx_amplitudes():
    """RX angle encoding picks up the imaginary unit on the |1> component."""
    features = [0.4, 1.1]

    qf = QuantumFloat(len(features))
    angle_encoding(qf, features, axis="x")

    expected = np.array([1.0], dtype=complex)
    for x in features:
        expected = np.kron(expected, np.array([np.cos(x / 2), -1j * np.sin(x / 2)]))

    assert np.allclose(statevector_array(qf), expected)


@pytest.mark.parametrize("num_qubits, features", [(3, [0.1, 0.2]), (2, [0.1, 0.2, 0.3])])
def test_angle_encoding_rejects_size_mismatch(num_qubits, features):
    """Angle encoding requires exactly one qubit per feature."""
    with pytest.raises(ValueError, match="one qubit per feature"):
        angle_encoding(QuantumFloat(num_qubits), features)


def test_angle_encoding_z_axis_phases():
    """
    Z angle encoding prepares the product state
    tensor_i (|0> + exp(i*x_i)|1>)/sqrt(2), up to a global phase.
    """
    features = [0.4, 1.1]

    qf = QuantumFloat(len(features))
    angle_encoding(qf, features, axis="z", prepare_basis=True)

    expected = np.array([1.0], dtype=complex)
    for x in features:
        expected = np.kron(expected, np.array([1.0, np.exp(1j * x)]) / np.sqrt(2))

    sv = statevector_array(qf)
    global_phase = sv[0] / expected[0]
    assert np.isclose(abs(global_phase), 1.0)
    assert np.allclose(sv, global_phase * expected)


def test_angle_encoding_z_axis_without_basis_preparation_is_unobservable():
    """
    Without the basis preparation, RZ only adds a global phase to |0>, so
    different features yield the same physical state.
    """
    qf_a = QuantumFloat(1)
    angle_encoding(qf_a, [0.7], axis="z", prepare_basis=False)

    qf_b = QuantumFloat(1)
    angle_encoding(qf_b, [3.9], axis="z", prepare_basis=False)

    assert np.allclose(np.abs(statevector_array(qf_a)), np.abs(statevector_array(qf_b)))


def test_angle_encoding_z_axis_warns_only_when_unspecified():
    """The implicit basis preparation is announced, an explicit choice is not."""
    with pytest.warns(BasisPreparationWarning):
        angle_encoding(QuantumFloat(2), [0.1, 0.2], axis="z")

    with warnings.catch_warnings():
        warnings.simplefilter("error", BasisPreparationWarning)
        angle_encoding(QuantumFloat(2), [0.1, 0.2], axis="z", prepare_basis=True)
        angle_encoding(QuantumFloat(2), [0.1, 0.2], axis="z", prepare_basis=False)


@pytest.mark.parametrize("features, outcome", [([0.0, np.pi], 2), ([np.pi, 0.0], 1)])
def test_angle_encoding_traces(features, outcome):
    """Angle encoding is compatible with the Jasp tracing path."""

    @jaspify
    def main(x):
        qf = QuantumFloat(2)
        angle_encoding(qf, x, prepare_basis=False)
        return measure(qf)

    assert main(jnp.array(features)) == outcome

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
from qrisp.qml import Encoding


def statevector_array(qv):
    return np.array(qv.qs.statevector("array"), dtype=complex)


def amplitudes_by_state(qv):
    """The statevector reindexed so that entry i is the amplitude of |i>."""
    amplitudes = statevector_array(qv)
    num_qubits = len(qv)
    return amplitudes.reshape((2,) * num_qubits).transpose().reshape(-1)


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
    Encoding.angle(n_features=len(features)).apply(qf, features)

    # statevector("array") is indexed in reverse bit order, so there the first
    # feature is the most significant bit and the Kronecker product follows the
    # feature order. In the |i> labelling it is the other way round, which is
    # what amplitudes_by_state corrects for.
    expected = np.array([1.0], dtype=complex)
    for x in features:
        expected = np.kron(expected, np.array([np.cos(x / 2), np.sin(x / 2)]))

    assert np.allclose(statevector_array(qf), expected)


def test_angle_encoding_rx_amplitudes():
    """RX angle encoding picks up the imaginary unit on the |1> component."""
    features = [0.4, 1.1]

    qf = QuantumFloat(len(features))
    Encoding.angle(n_features=len(features), axis="x").apply(qf, features)

    expected = np.array([1.0], dtype=complex)
    for x in features:
        expected = np.kron(expected, np.array([np.cos(x / 2), -1j * np.sin(x / 2)]))

    assert np.allclose(statevector_array(qf), expected)


@pytest.mark.parametrize("n_features, features", [(3, [0.1, 0.2]), (2, [0.1, 0.2, 0.3])])
def test_encoding_rejects_a_sample_of_the_wrong_width(n_features, features):
    """A sample must hold exactly the feature count the encoder was configured for."""
    with pytest.raises(ValueError, match="configured for"):
        Encoding.angle(n_features=n_features).apply(QuantumFloat(n_features), features)


def test_angle_encoding_z_axis_phases():
    """
    Z angle encoding prepares the product state
    tensor_i (|0> + exp(i*x_i)|1>)/sqrt(2), up to a global phase.
    """
    features = [0.4, 1.1]

    qf = QuantumFloat(len(features))
    Encoding.angle(n_features=len(features), axis="z", prepare_basis=True).apply(qf, features)

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
    encoding = Encoding.angle(n_features=1, axis="z", prepare_basis=False)

    qf_a = QuantumFloat(1)
    encoding.apply(qf_a, [0.7])

    qf_b = QuantumFloat(1)
    encoding.apply(qf_b, [3.9])

    assert np.allclose(np.abs(statevector_array(qf_a)), np.abs(statevector_array(qf_b)))


def test_angle_encoding_z_axis_warns_only_when_unspecified():
    """The implicit basis preparation is announced, an explicit choice is not."""
    with pytest.warns(BasisPreparationWarning):
        Encoding.angle(n_features=2, axis="z").apply(QuantumFloat(2), [0.1, 0.2])

    with warnings.catch_warnings():
        warnings.simplefilter("error", BasisPreparationWarning)
        Encoding.angle(n_features=2, axis="z", prepare_basis=True).apply(QuantumFloat(2), [0.1, 0.2])
        Encoding.angle(n_features=2, axis="z", prepare_basis=False).apply(QuantumFloat(2), [0.1, 0.2])


@pytest.mark.parametrize(
    "features, num_qubits",
    [([1.0, 2.0, 3.0, 4.0], 2), ([1.0, 2.0, 3.0], 2), ([0.5, -1.5], 1)],
)
def test_amplitude_encoding_normalizes_and_pads(features, num_qubits):
    """
    Feature x_i becomes the amplitude of |i>, zero-padded up to 2**n and
    normalized to unit norm.
    """
    qf = QuantumFloat(num_qubits)
    Encoding.amplitude(n_features=len(features)).apply(qf, features)

    expected = np.zeros(2**num_qubits)
    expected[: len(features)] = features
    expected /= np.linalg.norm(expected)

    # statevector("array") is indexed in reverse bit order, so index i of the
    # feature vector is the amplitude of |i>, not of array entry i.
    assert np.allclose(amplitudes_by_state(qf), expected)


def test_amplitude_encoding_rejects_oversized_feature_vector():
    """A register of n qubits cannot hold more than 2**n features."""
    with pytest.raises(ValueError, match="which holds 4 amplitudes"):
        Encoding.amplitude(n_features=5).apply(QuantumFloat(2), [1.0, 2.0, 3.0, 4.0, 5.0])


def test_amplitude_encoding_without_padding_requires_exact_length():
    """pad=False opts out of zero-padding, so the length must match exactly."""
    with pytest.raises(ValueError, match="pad=True"):
        Encoding.amplitude(n_features=3, pad=False).apply(QuantumFloat(2), [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "features, num_qubits",
    [([1.1, 0.7], 1), ([1.1, 0.7, 0.4, 2.0], 2), ([1.1, 0.7, 0.4], 2)],
)
def test_dense_angle_encoding_amplitudes(features, num_qubits):
    """
    Each qubit carries two features, as cos(x_2i/2)|0> + e^(i x_2i+1)
    sin(x_2i/2)|1>. An odd feature count leaves the last qubit without a phase.
    """
    qf = QuantumFloat(num_qubits)
    Encoding.dense_angle(n_features=len(features)).apply(qf, features)

    single_qubit_states = []
    for i in range(num_qubits):
        theta = features[2 * i]
        phase = features[2 * i + 1] if 2 * i + 1 < len(features) else 0.0
        single_qubit_states.append(np.array([np.cos(theta / 2), np.exp(1j * phase) * np.sin(theta / 2)]))

    # Qubit 0 is the least significant bit, so the product runs from the last
    # qubit down to the first.
    expected = np.array([1.0], dtype=complex)
    for state in reversed(single_qubit_states):
        expected = np.kron(expected, state)

    assert np.allclose(amplitudes_by_state(qf), expected)


def test_dense_angle_encoding_rejects_oversized_feature_vector():
    """Dense angle encoding holds at most two features per qubit."""
    with pytest.raises(ValueError, match="two features per qubit"):
        Encoding.dense_angle(n_features=5).apply(QuantumFloat(2), [0.1, 0.2, 0.3, 0.4, 0.5])


@pytest.mark.parametrize("bits, outcome", [([1, 0, 1], 5), ([0, 0, 0], 0), ([1, 1, 1], 7)])
def test_basis_encoding_prepares_basis_state(bits, outcome):
    """The bit vector is encoded into the corresponding computational basis state."""
    qf = QuantumFloat(len(bits))
    Encoding.basis(n_features=len(bits)).apply(qf, bits)

    assert qf.get_measurement() == {outcome: 1.0}


@pytest.mark.parametrize(
    "features, threshold, outcome",
    [([1, 2, 0], 0.0, 3), ([0.8, -0.2, 0.9], 0.0, 5), ([0.2, 0.7, 0.9], 0.5, 6)],
)
def test_basis_encoding_binarizes_its_input(features, threshold, outcome):
    """
    Basis encoding coerces its input to bits rather than rejecting it, so a
    value above the threshold sets its qubit instead of being read as a zero.
    """
    qf = QuantumFloat(len(features))
    Encoding.basis(n_features=len(features), threshold=threshold).apply(qf, features)

    assert qf.get_measurement() == {outcome: 1.0}


@pytest.mark.parametrize("features, outcome", [([0.0, np.pi], 2), ([np.pi, 0.0], 1)])
def test_angle_encoding_traces(features, outcome):
    """Angle encoding is compatible with the Jasp tracing path."""

    encoding = Encoding.angle(n_features=2, prepare_basis=False)

    @jaspify
    def main(x):
        qf = QuantumFloat(2)
        encoding.apply(qf, x)
        return measure(qf)

    assert main(jnp.array(features)) == outcome


@pytest.mark.parametrize(
    "features, num_qubits",
    [([1.0, 0.0, 0.0], 2), ([1.0, 0.0, 0.0, 0.0], 2), ([1.0, 0.0, 0.0], 3)],
)
def test_amplitude_encoding_traces(features, num_qubits):
    """
    Amplitude encoding is compatible with the Jasp tracing path, including the
    padded cases. The feature vector is the first basis state, so the register
    is deterministic and the measurement is a value rather than a sample.
    """

    encoding = Encoding.amplitude(n_features=len(features))

    @jaspify
    def main(x):
        qf = QuantumFloat(num_qubits)
        encoding.apply(qf, x)
        return measure(qf)

    assert main(jnp.array(features)) == 0


def test_next_power_of_two_of_zero_is_one():
    """
    Zero amplitudes need no qubits to hold them, so the smallest power of two
    that is at least zero is one. The bit_length of -1 would give two.
    """
    from qrisp.qml.encodings import _next_power_of_two

    assert _next_power_of_two(0) == 1

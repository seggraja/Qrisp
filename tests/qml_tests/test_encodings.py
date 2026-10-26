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
from jax.tree_util import register_pytree_node_class

from qrisp import BasisPreparationWarning, QuantumFloat, h, measure, ry, s
from qrisp.jasp import jaspify, jrange, terminal_sampling
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


@pytest.mark.parametrize("constructor", [Encoding.angle, Encoding.amplitude, Encoding.dense_angle, Encoding.basis])
@pytest.mark.parametrize("n_features, features", [(3, [0.1, 0.2]), (2, [0.1, 0.2, 0.3])])
def test_encoding_rejects_a_sample_of_the_wrong_width_under_tracing(constructor, n_features, features):
    """
    The width check also runs while a program is being traced, which is where a
    mismatch would otherwise pass silently: a short feature vector is read past
    its end and Jax clamps the read to the last value rather than raising.
    """

    encoding = constructor(n_features=n_features)

    @jaspify
    def main(x):
        qf = QuantumFloat(n_features)
        encoding.apply(qf, x)
        return measure(qf)

    with pytest.raises(ValueError, match="configured for"):
        main(jnp.array(features, dtype=float))


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
        Encoding.amplitude(n_features=4, pad=False).apply(QuantumFloat(3), [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("container", [list, tuple, np.array])
def test_amplitude_encoding_without_padding_accepts_any_sequence(container):
    """pad=False accepts the same feature sequences as the padded default."""
    qf = QuantumFloat(2)
    Encoding.amplitude(n_features=4, pad=False).apply(qf, container([1.0, -2.0, 3.0, 4.0]))

    assert np.allclose(amplitudes_by_state(qf), np.array([1.0, -2.0, 3.0, 4.0]) / np.sqrt(30))


@pytest.mark.parametrize("pad", [True, False])
@pytest.mark.parametrize("features", [[1j, 0.0, -2.0, 1 - 1j], [0.0, 1j, 0.0, 0.0]])
def test_amplitude_encoding_keeps_complex_phases(features, pad):
    """
    Complex features keep their phases, and a purely imaginary vector is not
    mistaken for one of zero norm. The overlap ignores the global phase, which
    the state preparation is free to choose.
    """
    qf = QuantumFloat(2)
    Encoding.amplitude(n_features=4, pad=pad).apply(qf, features)

    expected = np.array(features) / np.linalg.norm(features)

    assert np.isclose(abs(np.vdot(expected, amplitudes_by_state(qf))), 1.0)


@pytest.mark.parametrize("phase_readout", [False, True])
def test_amplitude_encoding_keeps_complex_phases_when_traced(phase_readout):
    """
    The traced path keeps complex phases on a register wider than the features
    need. A Hadamard on the first qubit turns the real part of each relative
    phase into a probability, and an S gate before it the imaginary part.
    """
    features = np.array([1.0, 1j, -1.0, -1j, 2j])
    encoding = Encoding.amplitude(n_features=len(features))

    @terminal_sampling
    def main(x):
        qf = QuantumFloat(4)
        encoding.apply(qf, x)
        if phase_readout:
            s(qf[0])
        h(qf[0])
        return qf

    amplitudes = np.zeros(16, dtype=complex)
    amplitudes[: len(features)] = features / np.linalg.norm(features)

    pairs = amplitudes.reshape(-1, 2)
    if phase_readout:
        pairs = pairs * np.array([1, 1j])
    mixed = np.stack([pairs[:, 0] + pairs[:, 1], pairs[:, 0] - pairs[:, 1]], axis=1) / np.sqrt(2)
    expected = np.abs(mixed.ravel()) ** 2

    distribution = main(jnp.array(features))
    observed = np.array([distribution.get(i, 0.0) for i in range(16)])

    assert np.allclose(observed, expected)


def test_scaled_encoding_rejects_complex_features():
    """A feature range orders real values only, so it cannot scale complex ones."""
    encoding = Encoding.amplitude(n_features=2, data_range=(0.0, 1.0))

    with pytest.raises(ValueError, match="complex features"):
        encoding.apply(QuantumFloat(1), [1.0, 1j])


def test_scaled_encoding_rejects_complex_features_when_traced():
    """The complex dtype is known while tracing, so the rejection runs there too."""
    encoding = Encoding.amplitude(n_features=2, data_range=(0.0, 1.0))

    @jaspify
    def main(x):
        qf = QuantumFloat(1)
        encoding.apply(qf, x)
        return measure(qf)

    with pytest.raises(ValueError, match="complex features"):
        main(jnp.array([1.0, 1j]))


def test_amplitude_encoding_without_padding_rejects_a_feature_count_no_register_holds():
    """
    A register holds a power of two amplitudes, so without padding any other
    feature count fails for every register. It is rejected on construction,
    where the traced path, which cannot see the register width, is covered too.
    """
    with pytest.raises(ValueError, match="power of two"):
        Encoding.amplitude(n_features=3, pad=False)


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
    [
        ([1.0, -2.0, 3.0, 4.0], 2),
        ([1.0, -2.0, 3.0], 2),
        ([1.0, -2.0, 3.0, 4.0], 3),
        ([0.0, 0.0, 1.0, 0.0], 3),
        ([1.0, 2.0, -3.0, 4.0, 5.0], 4),
        ([3.0], 1),
        ([-3.0], 2),
    ],
)
def test_amplitude_encoding_traces(features, num_qubits):
    """
    Under tracing, feature x_i becomes the amplitude of |i> with its sign, also
    on a register wider than the features need. A Hadamard on the first qubit
    turns the relative sign of each neighbouring pair of amplitudes into a
    probability, so a sign error changes the distribution as well.
    """
    encoding = Encoding.amplitude(n_features=len(features))

    @terminal_sampling
    def main(x):
        qf = QuantumFloat(num_qubits)
        encoding.apply(qf, x)
        h(qf[0])
        return qf

    amplitudes = np.zeros(2**num_qubits)
    amplitudes[: len(features)] = features
    amplitudes /= np.linalg.norm(amplitudes)

    # The first qubit is the least significant bit, so |2j> and |2j+1> are the
    # pairs the Hadamard mixes.
    pairs = amplitudes.reshape(-1, 2)
    mixed = np.stack([pairs[:, 0] + pairs[:, 1], pairs[:, 0] - pairs[:, 1]], axis=1) / np.sqrt(2)
    expected = mixed.ravel() ** 2

    distribution = main(jnp.array(features))
    observed = np.array([distribution.get(i, 0.0) for i in range(2**num_qubits)])

    assert np.allclose(observed, expected)


def test_next_power_of_two_of_zero_is_one():
    """
    Zero amplitudes need no qubits to hold them, so the smallest power of two
    that is at least zero is one. The bit_length of -1 would give two.
    """
    from qrisp.qml.encodings import _next_power_of_two

    assert _next_power_of_two(0) == 1


@register_pytree_node_class
class DoubleAngleEncoding(Encoding):
    """
    A minimal user written encoding, built only from the documented hook.
    """

    def encode(self, qv, x):
        for i in jrange(qv.size):
            ry(2 * x[i], qv[i])
        return qv


def test_subclass_inherits_the_traced_feature_check():
    """
    The check on the base class is the reason for subclassing rather than
    writing a bare function. Under tracing Jax reads past the end of a short
    feature vector by reusing its last value, so without this the wrong state is
    prepared with no error.
    """
    encoding = DoubleAngleEncoding(n_features=2)

    @jaspify
    def main(x):
        qf = QuantumFloat(2)
        encoding.apply(qf, x)
        return measure(qf)

    with pytest.raises(ValueError, match="configured for 2"):
        main(jnp.zeros(5))


def test_subclass_receives_scaled_features():
    """
    apply is documented as counting and scaling the features before encode sees
    them, so a subclass inherits the scaling without implementing it. Scaling
    [0, 1] onto [0, pi/2] and doubling inside encode gives RY(pi) on the second
    qubit, leaving |10> with qubit 0 as the least significant bit.
    """
    encoding = DoubleAngleEncoding(n_features=2, scaling=(jnp.pi / 2, 0.0, (0.0, jnp.pi / 2)))

    @jaspify
    def main(x):
        qf = QuantumFloat(2)
        encoding.apply(qf, x)
        return measure(qf)

    assert main(jnp.array([0.0, 1.0])) == 2


@pytest.mark.parametrize("constructor", [Encoding.angle, Encoding.dense_angle])
def test_rotation_encodings_reject_a_feature_range_longer_than_a_turn(constructor):
    """
    These encodings read a feature as a rotation angle, so a range wider than a
    full turn maps the smallest and largest value onto the same state.
    """
    data = np.array([[0.0, 1.0], [1.0, 2.0]])

    with pytest.raises(ValueError, match="longer than 2\\*pi"):
        constructor(n_features=2, data=data, feature_range=(0.0, 10.0))


def test_amplitude_encoding_accepts_a_feature_range_of_any_width():
    """
    Amplitude encoding normalizes the features, so the width of the range never
    reaches the state and the rotation periodicity limit does not apply. The
    range is rejected only when it is empty, which no scaling can use.
    """
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    wide = Encoding.amplitude(n_features=2, data=data, feature_range=(0.0, 10.0))
    narrow = Encoding.amplitude(n_features=2, data=data, feature_range=(0.0, np.pi))

    # Not data[0], which is the minimum of every column and therefore scales to
    # the lower bound of the range, leaving a zero vector to normalize.
    sample = data[1]

    qf_wide = QuantumFloat(1)
    qf_narrow = QuantumFloat(1)
    wide.apply(qf_wide, sample)
    narrow.apply(qf_narrow, sample)

    # The two ranges differ by more than a factor of three, and normalization
    # removes the difference entirely.
    assert np.allclose(statevector_array(qf_wide), statevector_array(qf_narrow))

    with pytest.raises(ValueError, match="empty feature range"):
        Encoding.amplitude(n_features=2, data=data, feature_range=(1.0, 1.0))


def test_amplitude_encoding_reports_a_sample_a_scaling_zeroed():
    """
    A scaling maps the sample sitting at the minimum of every feature onto the
    lower bound of the feature range, so the default range starting at 0 turns
    an ordinary row of a training set into a zero vector. Without a check here
    it reaches the state preparation, which reports a statevector of zero norm
    and names neither the sample nor the scaling that produced it.
    """
    X = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 5.0]])
    encoding = Encoding.amplitude(n_features=2, data=X)

    with pytest.raises(ValueError, match="vanishing norm"):
        encoding.apply(QuantumFloat(1), X[0])

    # The check mirrors the tolerance the state preparation applies, so a vector
    # that is not exactly zero but still too small for it is reported here too,
    # rather than falling through to the message that names no features.
    with pytest.raises(ValueError, match="vanishing norm"):
        Encoding.amplitude(n_features=2).apply(QuantumFloat(1), [1e-12, 0.0])

    # The remedy the message suggests has to actually encode that sample.
    shifted = Encoding.amplitude(n_features=2, data=X, feature_range=(1.0, 2.0))
    assert shifted.apply(QuantumFloat(1), X[0]) is not None

    # Samples away from the minimum were never affected.
    assert encoding.apply(QuantumFloat(1), X[2]) is not None


def test_encoding_without_its_hook_reports_what_is_missing():
    """
    A subclass that implements nothing would otherwise apply no gates and return
    an untouched QuantumVariable, which looks like a working encoding of zeros.
    """

    class Incomplete(Encoding):
        pass

    with pytest.raises(NotImplementedError, match="encode"):
        Incomplete(n_features=2).apply(QuantumFloat(2), [0.1, 0.2])

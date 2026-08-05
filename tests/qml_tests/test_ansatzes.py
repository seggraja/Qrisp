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

from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest

from qrisp import QuantumFloat, measure
from qrisp.jasp import jaspify
from qrisp.qml import (
    angle_encoding,
    data_reuploading,
    hardware_efficient,
    hardware_efficient_size,
    real_amplitudes,
    real_amplitudes_size,
)


def statevector_array(qv):
    return np.array(qv.qs.statevector("array"), dtype=complex)


def count_gates(qv, name):
    return sum(1 for i in qv.qs.compile().transpile().data if i.op.name == name)


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


def test_hardware_efficient_entanglers_differ():
    """Each entangler produces a different state, so both are really applied."""
    w = np.linspace(0.1, 0.8, 8)

    states = []
    for entangler in ["cx", "cz"]:
        qf = QuantumFloat(4)
        hardware_efficient(qf, w, layers=1, entangler=entangler)
        states.append(statevector_array(qf))

    assert not np.allclose(*states)


@pytest.mark.parametrize(
    "w, kwargs, match",
    [
        (np.zeros(7), {"layers": 2}, "Expected 9 parameters"),
        (np.zeros(6), {"rotations": ("w",)}, "Unknown rotation axis"),
        (np.zeros(6), {"entangler": "swap"}, "Unknown entangler"),
    ],
)
def test_hardware_efficient_rejects_invalid_arguments(w, kwargs, match):
    """
    The parameter count is determined by the qubits, layers and rotations, and
    the rotation axes and entanglers are restricted to the supported gates.
    """
    with pytest.raises(ValueError, match=match):
        hardware_efficient(QuantumFloat(3), w, **kwargs)


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


def test_real_amplitudes_keeps_amplitudes_real():
    """
    RY and CX have real matrix representations, so the prepared state stays in
    the real subspace. This is the property the ansatz is named for.
    """
    w = np.linspace(0.1, 2.9, real_amplitudes_size(3, layers=2))

    qf = QuantumFloat(3)
    real_amplitudes(qf, w, layers=2)

    assert np.allclose(statevector_array(qf).imag, 0.0)


@pytest.mark.parametrize("num_qubits, extra", [(2, 0), (3, 1), (4, 1)])
def test_real_amplitudes_ring_closes_the_chain(num_qubits, extra):
    """
    The ring adds one entangler over the linear chain, except on two qubits
    where the closing pair is the one the chain already covers.
    """
    w = np.zeros(real_amplitudes_size(num_qubits))

    qf_linear = QuantumFloat(num_qubits)
    real_amplitudes(qf_linear, w, entanglement="linear")

    qf_ring = QuantumFloat(num_qubits)
    real_amplitudes(qf_ring, w, entanglement="ring")

    assert count_gates(qf_ring, "cx") == count_gates(qf_linear, "cx") + extra


@pytest.mark.parametrize(
    "w, kwargs, match",
    [
        (np.zeros(7), {"layers": 2}, "Expected 9 parameters"),
        (np.zeros(6), {"entanglement": "full"}, "Unknown entanglement"),
    ],
)
def test_real_amplitudes_rejects_invalid_arguments(w, kwargs, match):
    """
    The parameter count is determined by the qubits and layers, and the
    entanglement is restricted to the supported connectivities.
    """
    with pytest.raises(ValueError, match=match):
        real_amplitudes(QuantumFloat(3), w, **kwargs)


@pytest.mark.parametrize("entanglement, expected", [("linear", 7), ("ring", 6)])
def test_real_amplitudes_propagates_a_flip_along_the_entanglers(entanglement, expected):
    """
    A flip on the first qubit travels along the CX chain, and the ring's closing
    entangler carries it back onto the first qubit. Run traced, so this covers
    the Jasp path as well.
    """

    @jaspify
    def main(w):
        qf = QuantumFloat(3)
        real_amplitudes(qf, w, entanglement=entanglement)
        return measure(qf)

    w = jnp.zeros(real_amplitudes_size(3)).at[0].set(jnp.pi)

    # The chain sets q0, q1 and q2, giving |7>. The ring then adds CX(2,0),
    # which clears q0 again and leaves |6>.
    assert main(w) == expected


def test_data_reuploading_encodes_once_per_layer():
    """
    The features are re-encoded in every layer, so three layers of RY(x/3)
    accumulate to a single RY(x) on an ansatz that applies no rotation.
    """
    per_layer = real_amplitudes_size(1)

    qf = QuantumFloat(1)
    data_reuploading(qf, [np.pi / 3], np.zeros(3 * per_layer), per_layer, layers=3)

    # Three RY(pi/3) rotations compose to RY(pi), which flips the qubit.
    assert np.allclose(np.abs(statevector_array(qf)), [0.0, 1.0], atol=1e-9)


def test_data_reuploading_uses_a_distinct_parameter_block_per_layer():
    """
    Each layer consumes its own block of w, so permuting the blocks changes the
    circuit. A shared block would make the two states identical.
    """
    per_layer = real_amplitudes_size(2)
    first = np.concatenate([np.full(per_layer, 0.3), np.full(per_layer, 1.1)])
    second = np.concatenate([np.full(per_layer, 1.1), np.full(per_layer, 0.3)])

    qf_first = QuantumFloat(2)
    data_reuploading(qf_first, [0.4, 0.8], first, per_layer)

    qf_second = QuantumFloat(2)
    data_reuploading(qf_second, [0.4, 0.8], second, per_layer)

    assert not np.allclose(statevector_array(qf_first), statevector_array(qf_second))


def test_data_reuploading_rejects_parameter_count_mismatch():
    """The parameters are split evenly over the layers."""
    per_layer = real_amplitudes_size(2)

    with pytest.raises(ValueError, match="Expected 12 parameters"):
        data_reuploading(QuantumFloat(2), [0.1, 0.2], np.zeros(5), per_layer, layers=3)


@pytest.mark.parametrize(
    "kwargs, per_layer",
    [
        ({}, real_amplitudes_size(1)),
        (
            {"encoding": partial(angle_encoding, axis="y"), "ansatz": hardware_efficient},
            hardware_efficient_size(1),
        ),
    ],
)
def test_data_reuploading_traces(kwargs, per_layer):
    """
    The circuit is compatible with the Jasp tracing path, with the default
    encoding and ansatz as well as with injected ones.
    """

    @jaspify
    def main(x, w):
        qf = QuantumFloat(1)
        data_reuploading(qf, x, w, per_layer, layers=2, **kwargs)
        return measure(qf)

    # On a single qubit there is nothing to entangle, so with vanishing
    # parameters the two uploads of RY(pi/2) compose to RY(pi) and flip it.
    assert main(jnp.array([jnp.pi / 2]), jnp.zeros(2 * per_layer)) == 1

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

import jax.numpy as jnp

from qrisp import QuantumFloat
from qrisp.operators import Z
from qrisp.qml import Ansatz, Encoding, expectation, hardware_efficient_size
from qrisp.jasp import jaspify

# Coarse enough to keep the suite fast, tight enough that a wrong sign or a
# swapped entry cannot pass.
PRECISION = 0.02
TOLERANCE = 0.1

NUM_QUBITS = 2
LAYERS = 1

# The parameters are held at zero throughout, which makes every rotation layer
# the identity and every expectation value exactly +1 or -1. The entangling
# CX is still applied, so the quantum block is not the identity: it maps the
# encoded basis state |q0 q1> to |q0, q0 XOR q1>.
#
#   encoded angles     state       after CX     <Z0>  <Z1>
#   [0,  0]            |00>        |00>          +1    +1
#   [pi, 0]            |10>        |11>          -1    -1
#   [0,  pi]           |01>        |01>          +1    -1
#   [pi, pi]           |11>        |10>          -1    +1
#
# These were derived by running the block, not by inspection: the CX makes the
# naive guess ([pi,0] -> [-1,+1]) wrong.
QUANTUM_TRUTH_TABLE = [
    ([0.0, 0.0], [1.0, 1.0]),
    ([np.pi, 0.0], [-1.0, -1.0]),
    ([0.0, np.pi], [1.0, -1.0]),
    ([np.pi, np.pi], [-1.0, 1.0]),
]


def _quantum_block():
    """A two-observable readout, which is what a classical head consumes."""
    encoding = Encoding.angle(n_features=NUM_QUBITS)
    ansatz = Ansatz.hardware_efficient(num_qubits=NUM_QUBITS, layers=LAYERS)

    def state_prep(x, w):
        qv = QuantumFloat(NUM_QUBITS)
        encoding.apply(qv, x)
        ansatz.apply(qv, w)
        return qv

    return expectation(state_prep, [Z(0), Z(1)], precision=PRECISION)


def _zero_parameters():
    return jnp.zeros(hardware_efficient_size(NUM_QUBITS, LAYERS))


@pytest.mark.parametrize("x, expected", QUANTUM_TRUTH_TABLE)
def test_quantum_block_returns_one_entry_per_observable(x, expected):
    """
    The readout a classical head consumes is a vector, not a scalar. A summed
    operator would collapse [pi, pi] to a single 0 and lose both entries.
    """
    result = _quantum_block()(jnp.array(x), _zero_parameters())

    assert np.shape(result) == (NUM_QUBITS,)
    assert np.allclose(result, expected, atol=TOLERANCE)


def test_classical_layer_before_the_quantum_block():
    """
    A classical layer feeding the encoding is what lets a wide feature vector
    reach a narrow register, so the scaling it applies has to survive into the
    encoded angles.
    """
    quantum_block = _quantum_block()

    def model(raw, w_in, w):
        return quantum_block(raw * w_in, w)

    # raw * w_in puts pi on both qubits, which is the [pi, pi] row above.
    result = model(jnp.array([1.0, 1.0]), jnp.array([np.pi, np.pi]), _zero_parameters())

    assert np.allclose(result, [-1.0, 1.0], atol=TOLERANCE)


def test_classical_layer_after_the_quantum_block():
    """
    The vector readout feeds an ordinary jnp expression, so a model can put a
    trainable head on top of the quantum block.
    """
    quantum_block = _quantum_block()

    def model(x, w, w_out, b_out):
        return jnp.tanh(jnp.dot(quantum_block(x, w), w_out) + b_out)

    # The [0, 0] row gives [+1, +1], so the head computes tanh(1 + 1 + 0).
    result = model(
        jnp.array([0.0, 0.0]),
        _zero_parameters(),
        jnp.array([1.0, 1.0]),
        0.0,
    )

    assert np.isclose(result, np.tanh(2.0), atol=TOLERANCE)


def test_classical_quantum_classical_composes_end_to_end():
    """
    The full sandwich, which is the composition a hybrid model is built from.
    Each stage is checked above on its own; this pins that they chain.
    """
    quantum_block = _quantum_block()

    def model(raw, w_in, w, w_out, b_out):
        h = raw * w_in
        q = quantum_block(h, w)
        return jnp.tanh(jnp.dot(q, w_out) + b_out)

    # raw * w_in gives [pi, 0] -> [-1, -1] -> tanh(-1 - 1 + 0).
    result = model(
        jnp.array([1.0, 0.0]),
        jnp.array([np.pi, np.pi]),
        _zero_parameters(),
        jnp.array([1.0, 1.0]),
        0.0,
    )

    assert np.isclose(result, np.tanh(-2.0), atol=TOLERANCE)


def test_quantum_block_output_is_a_jax_array():
    """
    A hybrid model is only composable if the readout is an ordinary Jax value.
    A raw ``measure`` returns a Clbit, which is not a Jax type and breaks any
    transformation applied to the surrounding function.
    """
    result = _quantum_block()(jnp.array([0.0, 0.0]), _zero_parameters())

    assert isinstance(result, jnp.ndarray)
    assert jnp.issubdtype(result.dtype, jnp.floating)


def test_hybrid_composition_traces():
    """
    Composition has to survive tracing, since that is the mode a training run
    uses. The classical parts are jnp expressions either side of the quantum
    block, so the whole model goes through one ``jaspify``.
    """
    quantum_block = _quantum_block()

    @jaspify
    def model(raw, w_in, w, w_out):
        q = quantum_block(raw * w_in, w)
        return jnp.tanh(jnp.dot(q, w_out))

    result = model(
        jnp.array([1.0, 1.0]),
        jnp.array([np.pi, np.pi]),
        _zero_parameters(),
        jnp.array([1.0, 1.0]),
    )

    # The [pi, pi] row gives [-1, +1], which sums to 0.
    assert np.isclose(result, 0.0, atol=TOLERANCE)

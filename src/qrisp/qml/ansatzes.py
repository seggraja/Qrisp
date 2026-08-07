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

from qrisp.core.gate_application_functions import cx, cz, rx, ry, rz
from qrisp.jasp import check_for_tracing_mode, jrange
from qrisp.qml.encodings import angle_encoding

__all__ = [
    "data_reuploading",
    "hardware_efficient",
    "hardware_efficient_size",
    "real_amplitudes",
    "real_amplitudes_size",
]

_ROTATION_GATES = {"x": rx, "y": ry, "z": rz}

_ENTANGLING_GATES = {"cx": cx, "cz": cz}

_ENTANGLEMENTS = ("linear", "ring")


def hardware_efficient(qv, w, layers=1, rotations=("y",), entangler="cx"):
    r"""
    Applies a hardware efficient ansatz to a QuantumVariable.

    The ansatz consists of ``layers`` repetitions of a parameterized rotation
    layer followed by an entangling layer, closed by a final rotation layer:

    .. math::

        \left( \prod_{l=1}^{L} U_{\text{ent}} R(w_l) \right) R(w_0)

    Each rotation layer applies one rotation per qubit for every axis in
    ``rotations``, so a register of $n$ qubits uses
    $(L+1) \cdot n \cdot \lvert\text{rotations}\rvert$ parameters. The final
    rotation layer ensures that the output of the last entangling layer is
    rotated before it is measured.

    The entangling layer applies a two-qubit gate to each neighbouring pair of
    qubits in a linear chain, which keeps the ansatz shallow and executable on
    hardware with nearest-neighbour connectivity.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable to apply the ansatz to.
    w : list[float] or numpy.ndarray
        The parameters. Either a flat vector of
        $(L+1) \cdot n \cdot \lvert\text{rotations}\rvert$ entries or an array
        shaped ``(layers + 1, len(rotations), n)``. Both are read in the same
        order, so a vector can be passed to an optimizer unchanged.
    layers : int, optional
        The number of rotation and entangling layer repetitions. The default
        is ``1``.
    rotations : tuple[str], optional
        The rotation axes applied to each qubit in a rotation layer. The
        default is ``("y",)``, which keeps the amplitudes real.
    entangler : {'cx', 'cz'}, optional
        The two-qubit gate of the entangling layer. The default is ``'cx'``.

    Raises
    ------
    ValueError
        If the number of parameters does not match the ansatz.
    ValueError
        If ``rotations`` holds an axis other than ``'x'``, ``'y'`` or ``'z'``.
    ValueError
        If ``entangler`` is not ``'cx'`` or ``'cz'``.

    Examples
    --------

    We apply two layers of the ansatz to a :ref:`QuantumFloat`.

    ::

        import numpy as np
        from qrisp import QuantumFloat
        from qrisp.qml import hardware_efficient

        qf = QuantumFloat(3)
        w = np.random.uniform(0, 2 * np.pi, hardware_efficient_size(3, layers=2))
        hardware_efficient(qf, w, layers=2)

    Combined with an encoding, this forms the quantum part of a model.

    ::

        from qrisp.qml import angle_encoding

        def circuit(x, w):
            qf = QuantumFloat(3)
            angle_encoding(qf, x)
            hardware_efficient(qf, w, layers=2)
            return qf

    """
    for axis in rotations:
        if axis not in _ROTATION_GATES:
            raise ValueError(
                f"Unknown rotation axis '{axis}' for the hardware efficient ansatz. "
                f"Supported axes are {sorted(_ROTATION_GATES)}."
            )

    if entangler not in _ENTANGLING_GATES:
        raise ValueError(
            f"Unknown entangler '{entangler}' for the hardware efficient ansatz. "
            f"Supported entanglers are {sorted(_ENTANGLING_GATES)}."
        )

    if layers < 0:
        raise ValueError(
            f"Tried to apply a hardware efficient ansatz with a negative layer "
            f"count ({layers}). The parameter count check does not catch this, "
            "as a negative count expects no parameters and would leave the "
            "QuantumVariable untouched."
        )

    num_rotation_layers = layers + 1

    if not check_for_tracing_mode():
        expected = hardware_efficient_size(qv.size, layers, rotations)
        if np.size(w) != expected:
            raise ValueError(
                f"Tried to apply a hardware efficient ansatz with {np.size(w)} "
                f"parameters to a QuantumVariable of size {qv.size}. Expected "
                f"{expected} parameters for {layers} layers and rotations "
                f"{tuple(rotations)}."
            )

    w = _flatten_parameters(w)

    entangling_gate = _ENTANGLING_GATES[entangler]
    num_rotations = len(rotations)

    for layer in range(num_rotation_layers):
        for axis_index, axis in enumerate(rotations):
            rotation = _ROTATION_GATES[axis]
            # The parameters of one rotation layer are contiguous, so the axes
            # of a layer follow each other before the next layer begins.
            offset = (layer * num_rotations + axis_index) * qv.size
            for i in jrange(qv.size):
                rotation(w[offset + i], qv[i])

        if layer < layers:
            _entangling_layer(qv, entangling_gate, "linear")

    return qv


def real_amplitudes(qv, w, layers=1, entanglement="ring"):
    r"""
    Applies an ansatz of $R_Y$ rotations and CX entanglers to a QuantumVariable.

    The ansatz consists of ``layers`` repetitions of an $R_Y$ rotation layer
    followed by a CX entangling layer, closed by a final rotation layer:

    .. math::

        \left( \prod_{l=1}^{L} U_{\text{ent}} R_Y(w_l) \right) R_Y(w_0)

    A register of $n$ qubits uses $(L+1) \cdot n$ parameters.

    Because $R_Y$ and CX both have real matrix representations, every state this
    ansatz prepares from $\ket{0}^{\otimes n}$ has real amplitudes. It therefore
    explores only the real subspace of the Hilbert space, which is sufficient
    whenever the quantity of interest is an expectation value of a real
    observable, and halves the number of parameters a general rotation layer
    would need.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable to apply the ansatz to.
    w : list[float] or numpy.ndarray
        The parameters. Either a flat vector of $(L+1) \cdot n$ entries or an
        array shaped ``(layers + 1, n)``. Both are read in the same order, so a
        vector can be passed to an optimizer unchanged.
    layers : int, optional
        The number of rotation and entangling layer repetitions. The default
        is ``1``.
    entanglement : {'ring', 'linear'}, optional
        The connectivity of the entangling layer. ``'linear'`` applies a CX to
        each neighbouring pair of qubits, ``'ring'`` additionally closes the
        chain between the last and the first qubit. The default is ``'ring'``.

    Raises
    ------
    ValueError
        If the number of parameters does not match the ansatz.
    ValueError
        If ``entanglement`` is not ``'ring'`` or ``'linear'``.

    Examples
    --------

    We apply two layers of the ansatz to a :ref:`QuantumFloat`.

    ::

        import numpy as np
        from qrisp import QuantumFloat
        from qrisp.qml import real_amplitudes, real_amplitudes_size

        qf = QuantumFloat(3)
        w = np.random.uniform(0, 2 * np.pi, real_amplitudes_size(3, layers=2))
        real_amplitudes(qf, w, layers=2)

    Every amplitude of the resulting state is real.

    >>> np.allclose(np.array(qf.qs.statevector("array")).imag, 0)
    True

    """
    if entanglement not in _ENTANGLEMENTS:
        raise ValueError(
            f"Unknown entanglement '{entanglement}' for the real amplitudes "
            f"ansatz. Supported entanglements are {sorted(_ENTANGLEMENTS)}."
        )

    if layers < 0:
        raise ValueError(
            f"Tried to apply a real amplitudes ansatz with a negative layer "
            f"count ({layers}). The parameter count check does not catch this, "
            "as a negative count expects no parameters and would leave the "
            "QuantumVariable untouched."
        )

    if not check_for_tracing_mode():
        expected = real_amplitudes_size(qv.size, layers)
        if np.size(w) != expected:
            raise ValueError(
                f"Tried to apply a real amplitudes ansatz with {np.size(w)} "
                f"parameters to a QuantumVariable of size {qv.size}. Expected "
                f"{expected} parameters for {layers} layers."
            )

    w = _flatten_parameters(w)

    for layer in range(layers + 1):
        offset = layer * qv.size
        for i in jrange(qv.size):
            ry(w[offset + i], qv[i])

        if layer < layers:
            _entangling_layer(qv, cx, entanglement)

    return qv


def real_amplitudes_size(num_qubits, layers=1):
    """
    Returns the number of parameters of a :ref:`real_amplitudes` ansatz.

    Parameters
    ----------
    num_qubits : int
        The size of the QuantumVariable the ansatz is applied to.
    layers : int, optional
        The number of rotation and entangling layer repetitions. The default
        is ``1``.

    Returns
    -------
    int
        The number of parameters.

    Examples
    --------

    ::

        import numpy as np
        from qrisp.qml import real_amplitudes, real_amplitudes_size

        w = np.random.uniform(0, 2 * np.pi, real_amplitudes_size(3, layers=2))

    """
    return (layers + 1) * num_qubits


def data_reuploading(qv, x, w, ansatz_size, layers=2, encoding=None, ansatz=None):
    r"""
    Applies alternating encoding and ansatz layers to a QuantumVariable.

    The feature vector is encoded once per layer rather than only at the start
    of the circuit:

    .. math::

        \prod_{l=1}^{L} W(w_l) S(x)

    where $S$ is the encoding and $W$ the parameterized ansatz.

    Re-encoding the same features between parameterized layers raises the
    frequency spectrum the circuit can represent. A single encoding layer
    followed by an ansatz produces a function of the features whose Fourier
    spectrum is limited by the encoding alone, so no amount of added parameters
    widens it. With $L$ encoding layers the accessible spectrum grows with $L$,
    which is what allows a circuit on few qubits to fit a rich function.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable to apply the circuit to.
    x : list[float] or numpy.ndarray
        The feature vector, encoded once per layer.
    w : list[float] or numpy.ndarray
        The parameters of all ansatz layers, concatenated. A flat vector of
        ``layers * ansatz_size`` entries.
    ansatz_size : int
        The number of parameters ``ansatz`` uses on ``qv``, as returned by the
        size function accompanying the ansatz. It splits ``w`` into one block
        per layer.
    layers : int, optional
        The number of encoding and ansatz layer repetitions. The default is
        ``2``, the smallest number that re-uploads the features at all.
    encoding : callable, optional
        Applied as ``encoding(qv, x)`` at the start of each layer. The default
        is :ref:`angle_encoding`.
    ansatz : callable, optional
        Applied as ``ansatz(qv, w_layer)`` after each encoding. The default is
        :ref:`real_amplitudes` with a single layer.

    Raises
    ------
    ValueError
        If the number of parameters does not match the layers.

    Examples
    --------

    We re-upload a two-dimensional feature vector over three layers.

    ::

        import numpy as np
        from qrisp import QuantumFloat
        from qrisp.qml import data_reuploading, real_amplitudes_size

        qf = QuantumFloat(2)
        per_layer = real_amplitudes_size(2)
        w = np.random.uniform(0, 2 * np.pi, 3 * per_layer)
        data_reuploading(qf, [0.4, 1.2], w, per_layer, layers=3)

    A different encoding and ansatz are supplied together with its parameter
    count.

    ::

        from functools import partial
        from qrisp.qml import angle_encoding, hardware_efficient
        from qrisp.qml import hardware_efficient_size

        qf = QuantumFloat(2)
        per_layer = hardware_efficient_size(2)
        w = np.random.uniform(0, 2 * np.pi, 2 * per_layer)
        data_reuploading(
            qf,
            [0.4, 1.2],
            w,
            per_layer,
            encoding=partial(angle_encoding, axis="x"),
            ansatz=hardware_efficient,
        )

    """
    if encoding is None:
        encoding = angle_encoding

    if ansatz is None:
        ansatz = real_amplitudes

    if np.size(w) != layers * ansatz_size:
        raise ValueError(
            f"Tried to apply data reuploading with {np.size(w)} parameters over "
            f"{layers} layers. Expected {layers * ansatz_size} parameters, "
            f"{ansatz_size} for each layer."
        )

    w = _flatten_parameters(w)

    for layer in range(layers):
        encoding(qv, x)
        ansatz(qv, w[layer * ansatz_size : (layer + 1) * ansatz_size])

    return qv


def _entangling_layer(qv, entangling_gate, entanglement):
    """
    Applies a two-qubit gate to each neighbouring pair of qubits.

    For ``"ring"`` the chain is closed by an additional gate on the last and
    first qubit, which is omitted on a register of two qubits where it would
    duplicate the pair the linear chain already covers.
    """
    for i in jrange(qv.size - 1):
        entangling_gate(qv[i], qv[i + 1])

    if entanglement == "ring":
        # The closing gate runs once on more than two qubits and not at all
        # otherwise, where it would repeat the pair the chain already covered.
        # qv.size is a tracer, so the count drives a jrange rather than an if.
        for _ in jrange(jnp.minimum(jnp.maximum(qv.size - 2, 0), 1)):
            entangling_gate(qv[qv.size - 1], qv[0])


def _flatten_parameters(w):
    """
    Views the parameters as a flat vector.

    Accepts an array shaped ``(layers + 1, len(rotations), n)`` as well as a
    flat vector, so that both an explicitly shaped array and a vector coming
    straight from an optimizer are valid inputs.
    """
    if hasattr(w, "ravel"):
        return w.ravel()

    # A nested list describes the same shape as the equivalent array, so it is
    # flattened rather than kept as the two-dimensional array asarray builds.
    return jnp.asarray(w, dtype=float).ravel()


def hardware_efficient_size(num_qubits, layers=1, rotations=("y",)):
    """
    Returns the number of parameters of a :ref:`hardware_efficient` ansatz.

    Parameters
    ----------
    num_qubits : int
        The size of the QuantumVariable the ansatz is applied to.
    layers : int, optional
        The number of rotation and entangling layer repetitions. The default
        is ``1``.
    rotations : tuple[str], optional
        The rotation axes applied to each qubit in a rotation layer. The
        default is ``("y",)``.

    Returns
    -------
    int
        The number of parameters.

    Examples
    --------

    ::

        import numpy as np
        from qrisp.qml import hardware_efficient, hardware_efficient_size

        w = np.random.uniform(0, 2 * np.pi, hardware_efficient_size(3, layers=2))

    """
    return (layers + 1) * len(rotations) * num_qubits

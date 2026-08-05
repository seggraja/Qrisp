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

__all__ = ["hardware_efficient", "hardware_efficient_size"]

_ROTATION_GATES = {"x": rx, "y": ry, "z": rz}

_ENTANGLING_GATES = {"cx": cx, "cz": cz}


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
        shaped ``(layers + 1, len(rotations), n)``. A flat vector is reshaped
        internally, so it can be passed to an optimizer unchanged.
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
            for i in jrange(qv.size - 1):
                entangling_gate(qv[i], qv[i + 1])

    return qv


def _flatten_parameters(w):
    """
    Views the parameters as a flat vector.

    Accepts an array shaped ``(layers + 1, len(rotations), n)`` as well as a
    flat vector, so that both an explicitly shaped array and a vector coming
    straight from an optimizer are valid inputs.
    """
    if hasattr(w, "ravel"):
        return w.ravel()

    return jnp.asarray(w, dtype=float)


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

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

from qrisp.core.gate_application_functions import h, rx, ry, rz
from qrisp.jasp import check_for_tracing_mode, jlen, jrange
from qrisp.misc.exceptions import BasisPreparationWarning

__all__ = ["angle_encoding"]

_ROTATION_GATES = {"x": rx, "y": ry, "z": rz}

_AXES_REQUIRING_BASIS_PREPARATION = {"z"}


# Marks that the caller did not specify ``prepare_basis``, so that only the
# implicit default emits a warning.
_UNSPECIFIED = object()


def angle_encoding(qv, x, axis="y", prepare_basis=_UNSPECIFIED):
    r"""
    Encodes a feature vector into the rotation angles of a QuantumVariable.

    Each feature is encoded as the rotation angle of a single qubit, i.e. the
    $i$-th feature $x_i$ is applied as a rotation on the $i$-th qubit:

    .. math::

        \ket{0}^{\otimes n} \mapsto \bigotimes_{i=0}^{n-1} R_{\text{axis}}(x_i)\ket{0}

    For the default ``axis="y"`` this prepares the product state

    .. math::

        \bigotimes_{i=0}^{n-1}
        \left( \cos(x_i/2)\ket{0} + \sin(x_i/2)\ket{1} \right).

    Angle encoding requires one qubit per feature. In contrast to amplitude
    encoding (see :ref:`prepare`), the feature vector is **not** normalized:
    the raw values are used as angles. Since rotation angles are
    $2\pi$-periodic, features are typically rescaled to an interval of length
    $2\pi$ (e.g. $[0,\pi]$) before encoding. This rescaling is left to the
    caller.

    **Encoding around the Z axis**

    A rotation only changes a state that does not already lie on the rotation
    axis. As $\ket{0}$ lies on the Z axis, $R_Z$ leaves it invariant up to a
    global phase,

    .. math::

        R_Z(x)\ket{0} = e^{-ix/2}\ket{0},

    which is not observable. Encoding features into $R_Z$ rotations of a
    register in $\ket{0}$ would therefore discard them: every value of $x$
    yields the same physical state.

    For ``axis="z"`` a Hadamard gate is thus applied to each qubit first,
    moving the state off the rotation axis so that the features take effect:

    .. math::

        \ket{0} \xrightarrow{H} \frac{\ket{0}+\ket{1}}{\sqrt 2}
        \xrightarrow{R_Z(x)} \frac{\ket{0}+e^{ix}\ket{1}}{\sqrt 2}.

    Pass ``prepare_basis=False`` to omit the Hadamard. This is appropriate
    whenever the register is not in $\ket{0}$, for instance when chaining
    encodings, when re-uploading data onto an already rotated register, or when
    a Hadamard layer has been applied separately.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable to encode the features into. Must contain exactly
        one qubit per feature.
    x : list[float] or numpy.ndarray
        The feature vector. Must have length equal to the size of ``qv``.
    axis : {'x', 'y', 'z'}, optional
        The rotation axis. The default is ``'y'``, which maps real-valued
        features to real-valued amplitudes.
    prepare_basis : bool, optional
        Whether to apply a Hadamard gate to each qubit before the rotation.
        This is only relevant for ``axis="z"``: :math:`R_Z` acts trivially on
        :math:`\ket{0}` up to a global phase, so without a basis change the
        encoded features would have no observable effect. The default is to
        apply the basis change and emit a :class:`BasisPreparationWarning`.
        Pass ``prepare_basis=False`` to suppress both, e.g. when encoding into
        a register that is not in the :math:`\ket{0}` state.

    Raises
    ------
    ValueError
        If the length of ``x`` does not match the size of ``qv``.
    ValueError
        If ``axis`` is not ``'x'``, ``'y'`` or ``'z'``.

    Warns
    -----
    BasisPreparationWarning
        If ``axis="z"`` and ``prepare_basis`` was not specified explicitly,
        indicating that a Hadamard gate was inserted automatically.

    Examples
    --------

    We encode a two-dimensional feature vector into a :ref:`QuantumFloat`.

    ::

        import numpy as np
        from qrisp import QuantumFloat
        from qrisp.qml import angle_encoding

        qf = QuantumFloat(2)
        angle_encoding(qf, [0, np.pi])

    The first feature is $0$, leaving the first qubit in $\ket{0}$. The second
    feature is $\pi$, flipping the second qubit to $\ket{1}$. The resulting
    state is therefore $\ket{q_0=0, q_1=1}$:

    >>> qf.qs.statevector()
    |2>

    Encoding around the X axis instead:

    ::

        qf = QuantumFloat(2)
        angle_encoding(qf, [0, np.pi], axis="x")

    """
    if axis not in _ROTATION_GATES:
        raise ValueError(
            f"Unknown rotation axis '{axis}' for angle encoding. Supported axes are {sorted(_ROTATION_GATES)}."
        )

    if not check_for_tracing_mode() and jlen(x) != qv.size:
        raise ValueError(
            f"Tried to angle encode {jlen(x)} features into a QuantumVariable of "
            f"size {qv.size}. Angle encoding requires one qubit per feature."
        )

    needs_basis_preparation = axis in _AXES_REQUIRING_BASIS_PREPARATION
    basis_preparation_requested = prepare_basis is _UNSPECIFIED or prepare_basis

    if needs_basis_preparation and prepare_basis is _UNSPECIFIED:
        warnings.warn(
            f"Angle encoding around the '{axis}' axis applies a Hadamard gate to "
            "each qubit before the rotation. Without it the rotation would only "
            "introduce a global phase on the |0> state and the encoded features "
            "would have no observable effect. Pass prepare_basis=False to skip "
            "the basis change, for example when encoding into a register that is "
            "not in the |0> state.",
            category=BasisPreparationWarning,
            stacklevel=2,
        )

    rotation = _ROTATION_GATES[axis]
    apply_basis_preparation = needs_basis_preparation and basis_preparation_requested

    for i in jrange(qv.size):
        if apply_basis_preparation:
            h(qv[i])
        rotation(x[i], qv[i])

    return qv

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

import numpy as np

from qrisp.alg_primitives.state_preparation import prepare
from qrisp.core.gate_application_functions import h, p, rx, ry, rz
from qrisp.core.gate_application_functions import x as x_gate
from qrisp.environments import control
from qrisp.jasp import check_for_tracing_mode, jlen, jrange
from qrisp.misc.exceptions import BasisPreparationWarning

__all__ = [
    "angle_encoding",
    "amplitude_encoding",
    "dense_angle_encoding",
    "basis_encoding",
]

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


def amplitude_encoding(qv, x, pad=True):
    r"""
    Encodes a feature vector into the amplitudes of a QuantumVariable.

    The feature vector $x$ is encoded as the amplitudes of the computational
    basis states, i.e. the $i$-th feature is the amplitude of $\ket{i}$:

    .. math::

        \ket{0}^{\otimes n} \mapsto
        \sum_{i} \frac{x_i}{\lVert x \rVert} \ket{i}

    A register of $n$ qubits holds $2^n$ amplitudes, so amplitude encoding
    represents $2^n$ features in $n$ qubits. This exponential density is its
    main advantage over :ref:`angle_encoding`, which requires one qubit per
    feature.

    The feature vector is normalized internally, as a quantum state must have
    unit norm. Only the direction of $x$ is therefore encoded; its length is
    not recoverable from the resulting state.

    **Feature vectors that are not a power of two**

    A register of $n$ qubits holds exactly $2^n$ amplitudes, while feature
    vectors are rarely a power of two in length. By default the vector is
    zero-padded up to the size of ``qv``, so the trailing basis states carry
    zero amplitude. Padding does not change the relative amplitudes of the
    features, since normalization happens after it.

    Pass ``pad=False`` to require an exactly matching feature vector instead.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable to encode the features into. Holds $2^n$ features
        for $n$ qubits.
    x : list[float] or numpy.ndarray
        The feature vector. Must not have zero norm, as it is normalized.
    pad : bool, optional
        Whether to zero-pad the feature vector up to $2^n$ entries. The default
        is ``True``. If ``False``, the length of ``x`` must be exactly $2^n$.

    Raises
    ------
    ValueError
        If ``x`` holds more features than ``qv`` can represent.
    ValueError
        If ``pad`` is ``False`` and the length of ``x`` is not exactly $2^n$.

    Examples
    --------

    We encode a three-dimensional feature vector into a :ref:`QuantumFloat`.

    ::

        import numpy as np
        from qrisp import QuantumFloat
        from qrisp.qml import amplitude_encoding

        qf = QuantumFloat(2)
        amplitude_encoding(qf, [1, 2, 3])

    Two qubits hold four amplitudes, so the vector is padded to $(1,2,3,0)$ and
    normalized by $\sqrt{14}$, giving the feature $x_i$ as the amplitude of
    $\ket{i}$:

    >>> np.round(qf.qs.statevector("array"), 4)
    array([0.2673, 0.8018, 0.5345, 0.])

    Note that ``statevector("array")`` is indexed in reverse bit order, so the
    amplitude of $\ket{i}$ is not at index $i$ of that array.

    """
    num_amplitudes = 2**qv.size

    if not check_for_tracing_mode():
        if jlen(x) > num_amplitudes:
            raise ValueError(
                f"Tried to amplitude encode {jlen(x)} features into a "
                f"QuantumVariable of size {qv.size}, which holds "
                f"{num_amplitudes} amplitudes."
            )

        if not pad and jlen(x) != num_amplitudes:
            raise ValueError(
                f"Tried to amplitude encode {jlen(x)} features into a "
                f"QuantumVariable holding {num_amplitudes} amplitudes. Pass "
                "pad=True to zero-pad the feature vector."
            )

    if pad:
        x = np.pad(np.asarray(x, dtype=float), (0, num_amplitudes - jlen(x)))

    prepare(qv, x)

    return qv


def dense_angle_encoding(qv, x):
    r"""
    Encodes two features per qubit into a rotation angle and a relative phase.

    Consecutive pairs of features are encoded into one qubit each: the feature
    $x_{2i}$ as a $Y$ rotation and the feature $x_{2i+1}$ as a relative phase,

    .. math::

        \ket{0} \mapsto
        \cos(x_{2i}/2)\ket{0} + e^{ix_{2i+1}}\sin(x_{2i}/2)\ket{1}.

    This halves the number of qubits required compared to
    :ref:`angle_encoding`, which places one feature per qubit. A register of
    $n$ qubits therefore holds $2n$ features.

    A feature vector of odd length leaves the last qubit without a phase
    feature, so only its rotation is applied.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable to encode the features into. Holds two features per
        qubit.
    x : list[float] or numpy.ndarray
        The feature vector. Must hold at most two features per qubit of ``qv``.

    Raises
    ------
    ValueError
        If ``x`` holds more than two features per qubit of ``qv``.

    Examples
    --------

    We encode a four-dimensional feature vector into two qubits.

    ::

        import numpy as np
        from qrisp import QuantumFloat
        from qrisp.qml import dense_angle_encoding

        qf = QuantumFloat(2)
        dense_angle_encoding(qf, [np.pi / 2, np.pi, np.pi / 2, 0])

    """
    if not check_for_tracing_mode() and jlen(x) > 2 * qv.size:
        raise ValueError(
            f"Tried to dense angle encode {jlen(x)} features into a "
            f"QuantumVariable of size {qv.size}. Dense angle encoding holds two "
            "features per qubit."
        )

    num_rotations = (jlen(x) + 1) // 2

    for i in jrange(num_rotations):
        ry(x[2 * i], qv[i])

    for i in jrange(jlen(x) // 2):
        p(x[2 * i + 1], qv[i])

    return qv


def basis_encoding(qv, x):
    r"""
    Encodes a vector of bits into a computational basis state.

    The $i$-th bit is encoded into the $i$-th qubit, preparing the basis state

    .. math::

        \ket{x_0 x_1 \dotsm x_{n-1}}.

    Basis encoding requires one qubit per bit and represents the feature vector
    exactly, without normalization or rescaling. In contrast to
    :ref:`angle_encoding` and :ref:`amplitude_encoding`, the features must be
    binary.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable to encode the bits into. Must contain exactly one
        qubit per bit.
    x : list[int] or numpy.ndarray
        The bit vector. Must have length equal to the size of ``qv`` and hold
        only the values 0 and 1.

    Raises
    ------
    ValueError
        If the length of ``x`` does not match the size of ``qv``.
    ValueError
        If ``x`` holds a value other than 0 or 1.

    Examples
    --------

    We encode a bit vector into a :ref:`QuantumFloat`.

    ::

        from qrisp import QuantumFloat
        from qrisp.qml import basis_encoding

        qf = QuantumFloat(3)
        basis_encoding(qf, [1, 0, 1])

    >>> qf.qs.statevector()
    |5>

    """
    if not check_for_tracing_mode():
        if jlen(x) != qv.size:
            raise ValueError(
                f"Tried to basis encode {jlen(x)} bits into a QuantumVariable of "
                f"size {qv.size}. Basis encoding requires one qubit per bit."
            )

        if not all(bit in (0, 1) for bit in x):
            raise ValueError("Basis encoding requires a vector of bits, i.e. of the values 0 and 1.")

    for i in jrange(qv.size):
        with control(x[i] == 1):
            x_gate(qv[i])

    return qv

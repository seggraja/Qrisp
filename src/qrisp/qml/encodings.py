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
from jax.tree_util import register_pytree_node_class, tree_leaves

from qrisp.alg_primitives.state_preparation import prepare
from qrisp.core.gate_application_functions import h, p, rx, ry, rz
from qrisp.core.gate_application_functions import x as x_gate
from qrisp.environments import control
from qrisp.jasp import check_for_tracing_mode, jlen, jrange
from qrisp.misc.exceptions import BasisPreparationWarning
from qrisp.qml.preprocessing import _check_finite

__all__ = ["Encoding"]

_ROTATION_GATES = {"x": rx, "y": ry, "z": rz}

_AXES_REQUIRING_BASIS_PREPARATION = {"z"}


class _Unspecified:
    # Pickles as a reference to the module-level instance below, so that an
    # encoding saved and loaded again still recognises it by identity.
    def __reduce__(self):
        return "_UNSPECIFIED"


# Marks that the caller did not specify ``prepare_basis``, so that only the
# implicit default emits a warning.
_UNSPECIFIED = _Unspecified()


def _angle_encoding(qv, x, axis="y", prepare_basis=_UNSPECIFIED):
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


def _amplitude_encoding(qv, x, pad=True):
    r"""
    Encodes a feature vector into the amplitudes of a QuantumVariable.

    The feature vector $x$ is encoded as the amplitudes of the computational
    basis states, i.e. the $i$-th feature is the amplitude of $\ket{i}$:

    .. math::

        \ket{0}^{\otimes n} \mapsto
        \sum_{i} \frac{x_i}{\lVert x \rVert} \ket{i}

    A register of $n$ qubits holds $2^n$ amplitudes, so amplitude encoding
    represents $2^n$ features in $n$ qubits. This exponential density is its
    main advantage over angle encoding, which requires one qubit per
    feature.

    The feature vector is normalized internally, as a quantum state must have
    unit norm. Only the direction of $x$ is therefore encoded; its length is
    not recoverable from the resulting state.

    Complex features are encoded with their phases. A phase shared by every
    feature is a global phase of the state and is not observable, so $x$ and
    $e^{i\varphi}x$ encode the same state, as do $x$ and $-x$.

    **Feature vectors that are not a power of two**

    A register of $n$ qubits holds exactly $2^n$ amplitudes, while feature
    vectors are rarely a power of two in length. By default the vector is
    zero-padded up to the size of ``qv``, so the trailing basis states carry
    zero amplitude. Padding does not change the relative amplitudes of the
    features, since normalization happens after it.

    A register larger than the features require is padded out in the same way,
    leaving the surplus basis states empty.

    Pass ``pad=False`` to require an exactly matching feature vector instead.

    Parameters
    ----------
    qv : QuantumVariable
        The QuantumVariable to encode the features into, in $\ket{0}$. Holds
        $2^n$ features for $n$ qubits.
    x : list[float] or list[complex] or numpy.ndarray
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

    """
    tracing = check_for_tracing_mode()
    x = jnp.asarray(x)
    x = x.astype(jnp.promote_types(x.dtype, jnp.float64))

    if not tracing:
        num_amplitudes = 2**qv.size

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

    if not tracing:
        # prepare() reports a vector it cannot normalize as a statevector of
        # zero norm, which names neither the features nor the scaling that
        # produced them. A scaled encoder reaches it for the sample sitting at
        # the minimum of every feature, which is an ordinary row of a training
        # set rather than a degenerate input.
        if not np.any(np.asarray(x)):
            raise ValueError(
                "Tried to amplitude encode a feature vector of vanishing norm, "
                "which has no direction to encode. A scaling maps the sample at "
                "the minimum of every feature onto the lower bound of the feature "
                "range, so a feature_range starting at 0 turns that sample into "
                "such a vector. Pass a feature_range with a positive lower bound, "
                "such as the default (1, 10), to keep it encodable."
            )

    # The state encodes only the direction of x, which dividing by its largest
    # magnitude keeps. Without it, a vector of tiny entries falls below the
    # tolerances of the state preparation, and one of huge entries overflows
    # its norm, both of which encode a wrong state.
    largest = jnp.max(jnp.abs(x))
    x = x / jnp.where(largest > 0, largest, 1.0)

    if pad:
        num_features = jnp.shape(x)[0]

        if tracing:
            # qv.size is a tracer and cannot size an allocation, so the target
            # comes from the feature count, which is known while tracing.
            num_amplitudes = _next_power_of_two(num_features)

        x = jnp.pad(x, (0, num_amplitudes - num_features))

    if not tracing:
        prepare(qv, x)
        return qv

    # prepare() reads the register width from qv.size, which under tracing does
    # not match a vector sized from the features. Preparing only the qubits the
    # vector fills leaves the remaining qubits in |0>, which in little endian
    # order is the zero padding up to the full register.
    num_qubits = jnp.shape(x)[0].bit_length() - 1

    # A single amplitude is the |0> state up to a global phase.
    if num_qubits > 0:
        prepare(qv[:num_qubits], x)

    return qv


def _next_power_of_two(n):
    """
    Returns the smallest power of two that is at least ``n``.
    """
    if n <= 1:
        return 1

    return 1 << (n - 1).bit_length()


def _dense_angle_encoding(qv, x):
    r"""
    Encodes two features per qubit into a rotation angle and a relative phase.

    Consecutive pairs of features are encoded into one qubit each: the feature
    $x_{2i}$ as a $Y$ rotation and the feature $x_{2i+1}$ as a relative phase,

    .. math::

        \ket{0} \mapsto
        \cos(x_{2i}/2)\ket{0} + e^{ix_{2i+1}}\sin(x_{2i}/2)\ket{1}.

    This halves the number of qubits required compared to
    angle encoding, which places one feature per qubit. A register of
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


def _basis_encoding(qv, x):
    r"""
    Encodes a vector of bits into a computational basis state.

    The $i$-th bit is encoded into the $i$-th qubit, preparing the basis state

    .. math::

        \ket{x_0 x_1 \dotsm x_{n-1}}.

    Basis encoding requires one qubit per bit and represents the feature vector
    exactly, without normalization or rescaling. In contrast to
    angle and amplitude encoding, the features must be
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


def _derive_range(data, data_range, feature_range, periodic=True):
    """
    Returns the affine map onto ``feature_range`` as a (scale, offset) pair.

    Runs outside the traced region, where every value is concrete, so the pair
    it returns are plain floats that the encoder carries into the trace.

    ``periodic`` marks the encodings that read a feature as a rotation angle,
    which are the ones a range longer than a full turn would wrap. Amplitude
    encoding normalizes its features instead, so the width of the range makes
    no difference to the state it prepares and is not restricted.
    """
    if data is not None and data_range is not None:
        raise ValueError(
            "Tried to configure an encoder with both data and data_range. Pass "
            "data to derive the range from a dataset, or data_range to state it "
            "directly, but not both."
        )

    low, high = feature_range

    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"Tried to scale into the feature range ({low}, {high}), which is not finite.")

    if not high > low:
        raise ValueError(f"Tried to scale into an empty feature range ({low}, {high}).")

    if periodic and high - low > 2 * np.pi:
        raise ValueError(
            f"Tried to scale into the feature range ({low}, {high}), which is "
            "longer than 2*pi. Rotation angles are 2*pi-periodic, so its "
            "endpoints would encode to the same state."
        )

    if data is not None:
        data = np.asarray(data, dtype=float)

        if data.ndim != 2:
            raise ValueError(
                f"Tried to derive a scaling range from data shaped {data.shape}. "
                "The range is taken across samples, so the data must be a "
                "two-dimensional array shaped (n_samples, n_features)."
            )

        _check_finite(data, "scale")
        minimum = data.min(axis=0)
        maximum = data.max(axis=0)
    else:
        minimum, maximum = (np.asarray(bound, dtype=float) for bound in data_range)

        if not (np.all(np.isfinite(minimum)) and np.all(np.isfinite(maximum))):
            raise ValueError(
                f"Tried to configure an encoder with the data range ({minimum}, {maximum}), which is not finite."
            )

        if not np.all(maximum >= minimum):
            raise ValueError(f"Tried to configure an encoder with the empty data range ({minimum}, {maximum}).")

    spread = maximum - minimum

    # A constant feature spans nothing to scale onto the range, so it is mapped
    # to the lower bound instead of dividing by zero.
    constant = spread == 0
    spread = np.where(constant, 1.0, spread)

    scale = np.where(constant, 0.0, (high - low) / spread)
    offset = low - scale * minimum

    return scale, offset


class Encoding:
    r"""
    A configured feature encoder.

    An encoder is built once per dataset and called once per sample. The
    classmethods :meth:`angle`, :meth:`amplitude`, :meth:`dense_angle` and
    :meth:`basis` each return one, configured for a given number of features:

    ::

        from qrisp import QuantumFloat
        from qrisp.jasp import jaspify
        from qrisp.qml import Encoding

        encoding = Encoding.angle(n_features=2, data=X)

        @jaspify
        def circuit(x):
            qf = QuantumFloat(2)
            encoding.apply(qf, x)
            return qf

    **Why the feature count is configured up front**

    Under :ref:`Jasp <jasp>` tracing a QuantumVariable's size is a tracer rather
    than a number, so an encoding cannot compare it against the features it was
    handed. Without a check, Jax reads past the end of a short feature vector by
    reusing its last value, which produces a plausible looking state rather than
    an error.

    Recording ``n_features`` while it is still an ordinary integer closes that
    gap: the number crosses into the traced region intact, and the length of
    each sample is available there as a static shape. The comparison therefore
    happens at trace time, before a single gate is emitted, and costs nothing
    while the program runs.

    **Register width is the caller's responsibility**

    The encoder validates the features it is given. It cannot validate the
    register, whose size is the tracer described above. A register wider than
    ``n_features`` silently reuses the last feature on the surplus qubits, and
    no error is raised. Keeping ``n_features`` and the register in step is left
    to the caller.

    **Optional preprocessing**

    Passing ``data`` or ``data_range`` bundles the scaling each encoding expects
    into the call, so that features are scaled and encoded in one step. Passing
    neither encodes the features exactly as given, for callers whose data is
    already scaled. The feature count is checked either way.

    The scaling is derived once, from the dataset rather than from each sample,
    so training and test data are placed on the same scale. Only the derived
    range is retained; ``data`` itself is not.

    **Writing an encoding**

    Subclassing this class gives an encoding of your own the same feature count
    check, scaling and error messages as the built-in ones. Implement
    :meth:`encode`, which receives features that have already been counted and
    scaled:

    ::

        from jax.tree_util import register_pytree_node_class
        from qrisp import ry
        from qrisp.jasp import jrange
        from qrisp.qml import Encoding

        @register_pytree_node_class
        class DoubleAngleEncoding(Encoding):
            def encode(self, qv, x):
                for i in jrange(qv.size):
                    ry(2 * x[i], qv[i])
                return qv

        encoding = DoubleAngleEncoding(n_features=2)

    Registering the class as a PyTree lets it be passed as an argument to a
    traced function rather than only captured from the enclosing scope. Settings
    of your own are kept across that roundtrip by returning them from
    ``_options``; the rebuilt encoding does not pass through ``__init__``, so a
    subclass may give it any signature.

    """

    # Whether the encoding is defined only on a register in |0>, as a state
    # preparation is, rather than as gates that act on any state.
    _from_zero_state = False

    def __init__(self, n_features, scaling=None):
        if not isinstance(n_features, (int, np.integer)) or isinstance(n_features, bool):
            raise ValueError(f"Tried to configure an encoder for {n_features!r} features, which is not an integer.")

        if n_features < 1:
            raise ValueError(f"Tried to configure an encoder for {n_features} features.")

        # A range for a different number of features would only fail to
        # broadcast once a sample arrives, or broadcast silently from a single
        # column onto every feature.
        if scaling is not None and np.shape(scaling[0]) not in ((), (n_features,)):
            raise ValueError(
                f"Tried to configure an encoder for {n_features} features with a "
                f"scaling range for {np.size(scaling[0])}."
            )

        self.n_features = int(n_features)
        self.scaling = scaling

    @classmethod
    def angle(
        cls, n_features, data=None, data_range=None, feature_range=(0.0, np.pi), axis="y", prepare_basis=_UNSPECIFIED
    ):
        r"""
        Returns an encoder placing one feature in each qubit's rotation angle.

        Encoding requires one qubit per feature. See :func:`minmax_scale
        <qrisp.qml.minmax_scale>` for why the features are scaled into an
        interval shorter than a full turn.

        Parameters
        ----------
        n_features : int
            The number of features each sample holds.
        data : numpy.ndarray, optional
            Features shaped ``(n_samples, n_features)`` to derive the scaling
            range from. Mutually exclusive with ``data_range``. The default is
            ``None``, in which case no scaling is applied.
        data_range : tuple, optional
            The ``(minimum, maximum)`` of the features, stated directly instead
            of derived from data. Either bound may be a scalar or one value per
            feature. Mutually exclusive with ``data``.
        feature_range : tuple[float, float], optional
            The interval to scale into, if a scaling was requested. The default
            is ``(0, numpy.pi)``.
        axis : {'x', 'y', 'z'}, optional
            The rotation axis. The default is ``'y'``, which maps real-valued
            features to real-valued amplitudes.
        prepare_basis : bool, optional
            Whether to apply a Hadamard gate to each qubit before the rotation.
            This is only relevant for ``axis="z"``: :math:`R_Z` acts trivially
            on :math:`\ket{0}` up to a global phase, so without a basis change
            the encoded features would have no observable effect. The default is
            to apply the basis change and emit a
            :class:`BasisPreparationWarning`.

        Returns
        -------
        Encoding
            The configured encoder.

        Examples
        --------

        ::

            import numpy as np
            from qrisp import QuantumFloat
            from qrisp.qml import Encoding

            qf = QuantumFloat(2)
            encoding = Encoding.angle(n_features=2)
            encoding.apply(qf, [0, np.pi])

        The first feature is $0$, leaving the first qubit in $\ket{0}$. The
        second is $\pi$, flipping the second qubit to $\ket{1}$:

        >>> qf.qs.statevector()
        |2>

        """
        if axis not in _ROTATION_GATES:
            raise ValueError(
                f"Unknown rotation axis '{axis}' for angle encoding. Supported axes are {sorted(_ROTATION_GATES)}."
            )

        encoding = _AngleEncoding(n_features, _scaling(data, data_range, feature_range))
        encoding.axis = axis
        encoding.prepare_basis = prepare_basis

        return encoding

    @classmethod
    def amplitude(cls, n_features, data=None, data_range=None, feature_range=(1.0, 10.0), pad=True):
        r"""
        Returns an encoder placing the features in a register's amplitudes.

        A register of $n$ qubits holds $2^n$ amplitudes, so this encoding
        represents $2^n$ features in $n$ qubits. The features are normalized
        internally, as a quantum state has unit norm, so only their direction is
        encoded. Complex features are encoded with their phases, provided no
        scaling is configured.

        **What normalization keeps**

        Normalization discards the magnitude of a feature vector: $x$ and $cx$
        encode the same state for every $c \neq 0$. Two samples whose features
        are proportional are therefore indistinguishable after encoding. Where
        the magnitude matters, it has to be carried by a feature of its own.

        **The default feature range**

        A scaling maps the sample at the minimum of every feature onto the
        lower bound of ``feature_range``. A lower bound of $0$ turns that sample
        into the zero vector, which has no direction to encode, and makes the
        samples near it unstable: two samples differing only slightly near the
        minimum can encode to clearly different states. The default range
        therefore starts above zero, so every scaled sample has a direction and
        nearby samples encode to nearby states.

        Since only the direction is encoded, the ratio of the upper to the lower
        bound sets the contrast between samples, not the bounds themselves. A
        larger ratio separates the states of dissimilar samples further, but
        lets the largest features dominate the state, so that a feature at its
        minimum contributes little. The default ratio of $10$ keeps every
        feature at least a tenth of the largest.

        **The register must be in** $\ket{0}$

        The encoding prepares a state rather than rotating one, so it is defined
        only on a register in $\ket{0}^{\otimes n}$. Applied to any other state
        its result is not the encoded vector. For the same reason it cannot be
        the encoding of :meth:`Ansatz.data_reuploading`, whose later layers act
        on a register the earlier ones have changed.

        **Cost under tracing**

        Amplitude encoding builds an arbitrary state rather than one rotation
        per qubit, and the traced path falls back to a general state
        preparation. It is roughly two orders of magnitude slower per call than
        angle encoding, which dominates a training run: an optimization that
        takes seconds with angle encoding takes tens of minutes with this one.

        Parameters
        ----------
        n_features : int
            The number of features each sample holds.
        data : numpy.ndarray, optional
            Features shaped ``(n_samples, n_features)`` to derive the scaling
            range from. Mutually exclusive with ``data_range``. The default is
            ``None``, in which case no scaling is applied.
        data_range : tuple, optional
            The ``(minimum, maximum)`` of the features, stated directly instead
            of derived from data. Mutually exclusive with ``data``.
        feature_range : tuple[float, float], optional
            The interval to scale into, if a scaling was requested. The default
            is ``(1, 10)``. Unlike the rotation based encodings this one accepts
            a range of any width, since the features are normalized rather than
            read as angles. A range containing zero can produce the zero
            vector, which is rejected when not tracing. Under tracing it cannot
            be reported and yields an undefined state.
        pad : bool, optional
            Whether to zero-pad the feature vector up to $2^n$ entries. The
            default is ``True``. If ``False``, ``n_features`` must be exactly
            $2^n$.

        Returns
        -------
        Encoding
            The configured encoder.

        Examples
        --------

        ::

            import numpy as np
            from qrisp import QuantumFloat
            from qrisp.qml import Encoding

            qf = QuantumFloat(2)
            encoding = Encoding.amplitude(n_features=3)
            encoding.apply(qf, [1, 2, 3])

        Two qubits hold four amplitudes, so the vector is padded to $(1,2,3,0)$
        and normalized by $\sqrt{14}$:

        >>> np.round(qf.qs.statevector("array"), 4)
        array([0.2673, 0.8018, 0.5345, 0.])

        """
        encoding = _AmplitudeEncoding(n_features, _scaling(data, data_range, feature_range, periodic=False))

        if not pad and _next_power_of_two(n_features) != n_features:
            raise ValueError(
                f"Tried to configure an amplitude encoder for {n_features} features "
                "with pad=False. A register holds a power of two amplitudes, so "
                "without padding the feature count must be one. Pass pad=True to "
                "zero-pad the feature vector."
            )

        encoding.pad = pad

        return encoding

    @classmethod
    def dense_angle(cls, n_features, data=None, data_range=None, feature_range=(0.0, np.pi)):
        r"""
        Returns an encoder placing two features in each qubit.

        Consecutive pairs of features are encoded into one qubit each, as a $Y$
        rotation and a relative phase, so a register of $n$ qubits holds $2n$
        features. An odd feature count leaves the last qubit without a phase.

        Parameters
        ----------
        n_features : int
            The number of features each sample holds.
        data : numpy.ndarray, optional
            Features shaped ``(n_samples, n_features)`` to derive the scaling
            range from. Mutually exclusive with ``data_range``. The default is
            ``None``, in which case no scaling is applied.
        data_range : tuple, optional
            The ``(minimum, maximum)`` of the features, stated directly instead
            of derived from data. Mutually exclusive with ``data``.
        feature_range : tuple[float, float], optional
            The interval to scale into, if a scaling was requested. The default
            is ``(0, numpy.pi)``.

        Returns
        -------
        Encoding
            The configured encoder.

        Examples
        --------

        ::

            import numpy as np
            from qrisp import QuantumFloat
            from qrisp.qml import Encoding

            qf = QuantumFloat(2)
            encoding = Encoding.dense_angle(n_features=4)
            encoding.apply(qf, [np.pi / 2, np.pi, np.pi / 2, 0])

        """
        return _DenseAngleEncoding(n_features, _scaling(data, data_range, feature_range))

    @classmethod
    def basis(cls, n_features, threshold=0.0):
        r"""
        Returns an encoder placing one bit in each qubit.

        The $i$-th bit is encoded into the $i$-th qubit, preparing the basis
        state $\ket{x_0 x_1 \dotsm x_{n-1}}$, so encoding requires one qubit per
        bit.

        **Features are binarized**

        Basis encoding is defined for the values 0 and 1 alone. Rather than
        rejecting anything else, this encoder binarizes what it is given: a
        feature above ``threshold`` becomes ``1`` and a feature at or below it
        becomes ``0``. Continuous features are therefore accepted, and reach the
        qubits reduced to bits rather than unchanged.

        Parameters
        ----------
        n_features : int
            The number of features each sample holds.
        threshold : float, optional
            The value a feature has to exceed to become ``1``. The default is
            ``0.0``.

        Returns
        -------
        Encoding
            The configured encoder.

        Examples
        --------

        ::

            from qrisp import QuantumFloat
            from qrisp.qml import Encoding

            qf = QuantumFloat(3)
            encoding = Encoding.basis(n_features=3)
            encoding.apply(qf, [1, 0, 1])

        >>> qf.qs.statevector()
        |5>

        """
        encoding = _BasisEncoding(n_features)
        encoding.threshold = threshold

        return encoding

    def apply(self, qv, x):
        """
        Encodes one sample into ``qv``.

        Parameters
        ----------
        qv : QuantumVariable
            The QuantumVariable to encode the features into. Its size is not
            validated; see the note on register width in the class docstring.
        x : list[float] or numpy.ndarray
            The feature vector, holding ``n_features`` features.

        Returns
        -------
        QuantumVariable
            The QuantumVariable the features were encoded into.

        Raises
        ------
        ValueError
            If ``x`` is not one-dimensional.
        ValueError
            If ``x`` does not hold exactly ``n_features`` features.
        ValueError
            If ``x`` is complex and the encoder was configured with a scaling.

        """
        # jnp.shape is static under tracing, so unlike qv.size these comparisons
        # also run inside a traced program, where the bugs they catch live.
        if jnp.ndim(jnp.asarray(x)) != 1:
            raise ValueError(
                f"Tried to encode a sample shaped {jnp.shape(x)}. A sample is a "
                "one-dimensional vector of features, so a batch is encoded one "
                "sample at a time."
            )

        num_features = jnp.shape(jnp.asarray(x))[0]

        if num_features != self.n_features:
            raise ValueError(
                f"Tried to encode {num_features} features with an encoder configured for {self.n_features}."
            )

        return self.encode(qv, self._prepare(x))

    def tree_flatten(self):
        """
        PyTree flatten for JAX.

        The configuration is auxiliary data rather than a child, so that
        ``n_features`` stays a plain integer that the trace time check can
        compare against. The scale and offset of a scaling are children: Jax
        compares auxiliary data for equality, which arrays do not support.
        """
        if self.scaling is None:
            return (), (self.n_features, None, self._options())

        scale, offset, feature_range = self.scaling

        return (scale, offset), (self.n_features, feature_range, self._options())

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        PyTree unflatten for JAX.
        """
        n_features, feature_range, options = aux_data

        # Built without __init__, whose signature a subclass may change. The
        # configuration was validated when the flattened encoding was built.
        encoding = cls.__new__(cls)
        encoding.n_features = n_features
        encoding.scaling = None if feature_range is None else (*children, feature_range)

        for name, value in options.items():
            setattr(encoding, name, value)

        return encoding

    def __setstate__(self, state):
        # A subclass defined in a script is rebuilt from its code when loaded
        # in another process, without running the decorator that registered it
        # as a PyTree. Unregistered, it is its own single leaf. treedef_is_leaf
        # cannot tell, as it also holds for a registered node without children.
        self.__dict__.update(state)
        leaves = tree_leaves(self)

        if len(leaves) == 1 and leaves[0] is self:
            register_pytree_node_class(type(self))

    def encode(self, qv, x):
        """
        Emits the gates encoding one sample.

        Subclasses implement this to define their encoding. It is called by
        :meth:`apply`, which has already rejected a feature vector of the wrong
        length and applied any configured scaling, so an implementation can
        rely on ``x`` holding exactly ``n_features`` scaled features.

        Parameters
        ----------
        qv : QuantumVariable
            The QuantumVariable to encode the features into. Under tracing its
            size is a tracer, so it cannot be compared against ``n_features``
            or used to size an allocation. Loop over it with
            :ref:`jrange <jrange>` rather than ``range(len(qv))``.
        x : list[float] or numpy.ndarray
            The feature vector, holding ``n_features`` features.

        Returns
        -------
        QuantumVariable
            The QuantumVariable the features were encoded into.

        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement encode, which an encoding needs in order to emit its gates."
        )

    def _options(self):
        """
        Returns the per-encoding settings, which subclasses set after building.
        """
        return {}

    def _prepare(self, x):
        """
        Applies the configured scaling, or returns the features untouched.
        """
        if self.scaling is None:
            return x

        if jnp.iscomplexobj(jnp.asarray(x)):
            raise ValueError(
                "Tried to scale complex features into a feature range. A range "
                "orders real values only, so complex features are encoded without "
                "a scaling: pass neither data nor data_range."
            )

        scale, offset, (low, high) = self.scaling

        # Clipping matches minmax_scale: a sample outside the configured range
        # would otherwise leave the interval the encoding expects.
        return jnp.clip(scale * jnp.asarray(x, dtype=float) + offset, low, high)


def _scaling(data, data_range, feature_range, periodic=True):
    """
    Returns the affine map onto ``feature_range``, or None if none was requested.
    """
    if data is None and data_range is None:
        return None

    scale, offset = _derive_range(data, data_range, feature_range, periodic=periodic)
    low, high = feature_range

    return scale, offset, (float(low), float(high))


@register_pytree_node_class
class _AngleEncoding(Encoding):
    def _options(self):
        return {"axis": self.axis, "prepare_basis": self.prepare_basis}

    def encode(self, qv, x):
        return _angle_encoding(qv, x, axis=self.axis, prepare_basis=self.prepare_basis)


@register_pytree_node_class
class _AmplitudeEncoding(Encoding):
    _from_zero_state = True

    def _options(self):
        return {"pad": self.pad}

    def encode(self, qv, x):
        return _amplitude_encoding(qv, x, pad=self.pad)


@register_pytree_node_class
class _DenseAngleEncoding(Encoding):
    def encode(self, qv, x):
        return _dense_angle_encoding(qv, x)


@register_pytree_node_class
class _BasisEncoding(Encoding):
    def _options(self):
        return {"threshold": self.threshold}

    def _prepare(self, x):
        # Coercing rather than rejecting keeps a stray value from being read as
        # a zero bit, which is what an out-of-range value would otherwise mean.
        return (jnp.asarray(x) > self.threshold).astype(int)

    def encode(self, qv, x):
        return _basis_encoding(qv, x)

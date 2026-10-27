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

__all__ = ["binarize", "minmax_scale", "normalize"]


def _check_finite(X, name):
    """
    Rejects features that are not finite.

    A NaN propagates through the scalers into every encoded angle, and
    comparisons against it are always False, so it would be encoded as a zero
    bit rather than reported. Neither is recoverable once the features reach a
    circuit, so they are rejected while the offending value can still be seen.
    """
    if not np.all(np.isfinite(X)):
        raise ValueError(f"Tried to {name} features that are not all finite, i.e. that hold a NaN or an infinity.")


def minmax_scale(X_train, X_test=None, feature_range=(0.0, np.pi), periodic=True):
    r"""
    Scales features into an interval suitable for an encoding.

    Each feature is mapped linearly from its range on the training data onto
    ``feature_range``:

    .. math::

        x' = a + (b - a) \frac{x - \min(x)}{\max(x) - \min(x)}

    The minimum and maximum are taken over ``X_train`` alone, and the same two
    numbers are applied to ``X_test``. Deriving them from the test data as well
    would let information about it reach the model during training, which makes
    the measured accuracy better than the model actually is.

    Because the test data is scaled with the training range, a test feature
    outside that range would leave ``feature_range``. Such values are clipped to
    its bounds, which keeps every encoded angle inside the interval the encoding
    expects.

    **Why the interval matters**

    :meth:`Encoding.angle` and :meth:`Encoding.dense_angle` read features as
    rotation angles. Rotations are $2\pi$-periodic, so features spanning more
    than $2\pi$ wrap: the values $x$ and $x + 2\pi$ prepare the same state and
    the model cannot separate them. The default range $[0, \pi]$ stays well
    inside one period and spans $\ket{0}$ to $\ket{1}$ for a $Y$ rotation.

    A range of exactly $2\pi$ is a full turn, so its two endpoints encode to the
    same state and the smallest and largest values of a feature become
    indistinguishable. Only the endpoints collide, and the range is accepted for
    callers who want the full period; the default $[0,\pi]$ avoids the collision
    entirely.

    **Amplitude encoding**

    :meth:`Encoding.amplitude` normalizes the features rather than reading them
    as angles, so the width of the range is not limited by a period. Pass
    ``periodic=False`` to lift the limit. Only the ratio of the bounds then
    reaches the state, and the lower bound should be positive: a range
    starting at $0$ maps the sample at the minimum of every feature to the zero
    vector, which has no direction to encode. The range ``(1, 10)`` matches the
    scaling :meth:`Encoding.amplitude` applies by default.

    A feature that is constant across the training data has a zero range. It
    carries no information, and is mapped to the lower bound of
    ``feature_range`` rather than producing a division by zero.

    Parameters
    ----------
    X_train : numpy.ndarray
        The training features, shaped ``(N, n_features)``. The scaling is
        derived from this array.
    X_test : numpy.ndarray, optional
        Further features to scale with the training range, shaped
        ``(M, n_features)``. The default is ``None``, in which case only the
        scaled training features are returned.
    feature_range : tuple[float, float], optional
        The interval to scale into. The default is ``(0, numpy.pi)``. A range
        longer than $2\pi$ raises a ``ValueError`` unless ``periodic`` is
        ``False``, since the rotation based encodings could not distinguish its
        endpoints.
    periodic : bool, optional
        Whether the features will be read as rotation angles, which limits
        ``feature_range`` to a width of $2\pi$. The default is ``True``. Pass
        ``False`` for amplitude encoding.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        The scaled training features, and the scaled test features if
        ``X_test`` was given.

    Raises
    ------
    ValueError
        If ``feature_range`` is empty or not finite, or spans more than $2\pi$
        while ``periodic`` is ``True``.
    ValueError
        If ``X_test`` does not have the same number of features as
        ``X_train``.

    Examples
    --------

    We scale a training and a test set together.

    ::

        import numpy as np
        from qrisp.qml import minmax_scale

        X_train = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
        X_test = np.array([[0.5, 15.0]])

        X_train_scaled, X_test_scaled = minmax_scale(X_train, X_test)

    The training features now span the full interval.

    >>> np.round(X_train_scaled, 4)
    array([[0.    , 0.    ],
           [1.5708, 1.5708],
           [3.1416, 3.1416]])

    The test features are placed on the same scale rather than on their own.

    >>> np.round(X_test_scaled, 4)
    array([[0.7854, 0.7854]])

    The scaled features are ready to be encoded.

    ::

        from qrisp import QuantumFloat
        from qrisp.qml import Encoding

        qf = QuantumFloat(2)
        Encoding.angle(n_features=2).apply(qf, X_train_scaled[0])

    """
    low, high = feature_range

    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"Tried to scale into the feature range ({low}, {high}), which is not finite.")

    if not high > low:
        raise ValueError(f"Tried to scale into an empty feature range ({low}, {high}).")

    if periodic and high - low > 2 * np.pi:
        raise ValueError(
            f"Tried to scale into the feature range ({low}, {high}), which is "
            f"longer than 2*pi. Rotation angles are 2*pi-periodic, so its "
            "endpoints would encode to the same state. Pass periodic=False if "
            "the features are amplitude encoded."
        )

    X_train = np.asarray(X_train, dtype=float)

    if X_train.ndim != 2:
        raise ValueError(
            f"Tried to scale features shaped {X_train.shape}. minmax_scale "
            "derives its range across samples, so it requires a two-dimensional "
            "array shaped (n_samples, n_features). A single sample has no range "
            "to scale onto the feature range."
        )

    _check_finite(X_train, "scale")

    minimum = X_train.min(axis=0)
    maximum = X_train.max(axis=0)
    spread = maximum - minimum

    # A constant feature spans nothing to scale onto the range, so it is mapped
    # to the lower bound instead of dividing by zero.
    constant = spread == 0
    spread = np.where(constant, 1.0, spread)

    def scale(X):
        X = np.asarray(X, dtype=float)
        _check_finite(X, "scale")

        if X.shape[-1] != X_train.shape[-1]:
            raise ValueError(
                f"Tried to scale {X.shape[-1]} features with a range derived from {X_train.shape[-1]} features."
            )

        scaled = low + (high - low) * (X - minimum) / spread
        scaled = np.where(constant, low, scaled)

        # Test features outside the training range would land outside
        # feature_range, where the encodings no longer behave as documented.
        return np.clip(scaled, low, high)

    if X_test is None:
        return scale(X_train)

    return scale(X_train), scale(X_test)


def normalize(X):
    r"""
    Scales each sample to unit norm for amplitude encoding.

    Every row is divided by its Euclidean norm:

    .. math::

        x' = \frac{x}{\lVert x \rVert}

    :meth:`Encoding.amplitude` writes the features into the amplitudes of a
    register, and a quantum state has unit norm. It therefore normalizes
    internally, which means the length of a feature vector never reaches the
    circuit: only its direction is encoded. Two samples differing by a scalar
    factor prepare the same state.

    Applying this function first makes that visible in the data rather than
    leaving it to happen silently inside the encoding. Where the magnitude
    carries information, it has to be kept as a separate feature, and this
    function shows what would otherwise be lost.

    Parameters
    ----------
    X : numpy.ndarray
        The features, shaped ``(N, n_features)``. A single sample shaped
        ``(n_features,)`` is also accepted.

    Returns
    -------
    numpy.ndarray
        The features, each sample scaled to unit norm.

    Raises
    ------
    ValueError
        If any sample has zero norm, which has no direction to encode.

    Examples
    --------

    ::

        import numpy as np
        from qrisp.qml import normalize

        X = np.array([[3.0, 4.0], [1.0, 0.0]])
        X_normalized = normalize(X)

    The first sample had norm $5$, so it is divided by it.

    >>> np.round(X_normalized, 4)
    array([[0.6, 0.8],
           [1. , 0. ]])

    Every sample now has unit norm, which is what the encoded state carries.

    >>> np.round(np.linalg.norm(X_normalized, axis=1), 4)
    array([1., 1.])

    """
    X = np.asarray(X, dtype=float)
    _check_finite(X, "normalize")

    # Dividing by the largest component first keeps the norm of a sample of very
    # small components from underflowing to zero, which would report a well
    # defined direction as having none.
    largest = np.max(np.abs(X), axis=-1, keepdims=True)

    if np.any(largest == 0):
        raise ValueError("Tried to normalize a sample of zero norm, which has no direction to encode into amplitudes.")

    rescaled = X / largest
    norms = np.linalg.norm(rescaled, axis=-1, keepdims=True)

    return rescaled / norms


def binarize(X, threshold=0.0):
    r"""
    Turns features into the bit values basis encoding requires.

    Each feature above ``threshold`` becomes ``1`` and each feature at or below
    it becomes ``0``:

    .. math::

        x' = \begin{cases} 1 & x > t \\ 0 & x \leq t \end{cases}

    :meth:`Encoding.basis` maps a bit vector onto a computational basis state and
    accepts only the values 0 and 1, so continuous features have to be reduced
    to bits before they can be encoded that way.

    This discards everything about a feature except which side of the threshold
    it falls on. It suits data that is already close to binary, and loses most
    of the information in data that is not.

    The default threshold of $0$ splits data centred on zero at its centre.
    Features scaled with :func:`minmax_scale` lie in its ``feature_range``,
    whose lower bound is $0$ by default, so this threshold turns every feature
    above its training minimum into ``1``. Split such features at the middle of
    the range instead, ``threshold=numpy.pi / 2`` for the default range.

    Parameters
    ----------
    X : numpy.ndarray
        The features, shaped ``(N, n_features)``. A single sample shaped
        ``(n_features,)`` is also accepted.
    threshold : float, optional
        The value a feature has to exceed to become ``1``. The default is
        ``0.0``.

    Returns
    -------
    numpy.ndarray
        The features as integers, holding only the values 0 and 1.

    Examples
    --------

    ::

        import numpy as np
        from qrisp.qml import binarize

        X = np.array([[0.8, -0.2], [-1.0, 2.0]])
        bits = binarize(X)

    >>> bits
    array([[1, 0],
           [0, 1]])

    The result is directly encodable.

    ::

        from qrisp import QuantumFloat
        from qrisp.qml import Encoding

        qf = QuantumFloat(2)
        Encoding.basis(n_features=2).apply(qf, bits[0])

    A threshold splits continuous features at a chosen value instead.

    >>> binarize(np.array([[0.2, 0.7]]), threshold=0.5)
    array([[0, 1]])

    """
    X = np.asarray(X, dtype=float)
    _check_finite(X, "binarize")

    return (X > threshold).astype(int)

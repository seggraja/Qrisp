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
from jax.tree_util import register_pytree_node_class, tree_leaves

from qrisp.core.gate_application_functions import cx, cz, rx, ry, rz
from qrisp.jasp import check_for_tracing_mode, jrange

__all__ = [
    "Ansatz",
    "hardware_efficient_size",
    "real_amplitudes_size",
]

_ROTATION_GATES = {"x": rx, "y": ry, "z": rz}

_ENTANGLING_GATES = {"cx": cx, "cz": cz}

_ENTANGLEMENTS = ("linear", "ring")


def _hardware_efficient(qv, w, layers=1, rotations=("y",), entangler="cx"):
    r"""
    Applies a hardware efficient ansatz to a QuantumVariable.

    The ansatz consists of ``layers`` repetitions of a parameterized rotation
    layer followed by an entangling layer, closed by a final rotation layer:

    .. math::

        R(w_L)\, U_{\text{ent}} \cdots R(w_1)\, U_{\text{ent}}\, R(w_0)

    The operators act from right to left, so $R(w_0)$ is applied first.

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


def _real_amplitudes(qv, w, layers=1, entanglement="ring"):
    r"""
    Applies an ansatz of $R_Y$ rotations and CX entanglers to a QuantumVariable.

    The ansatz consists of ``layers`` repetitions of an $R_Y$ rotation layer
    followed by a CX entangling layer, closed by a final rotation layer:

    .. math::

        R_Y(w_L)\, U_{\text{ent}} \cdots R_Y(w_1)\, U_{\text{ent}}\, R_Y(w_0)

    The operators act from right to left, so $R_Y(w_0)$ is applied first. A
    register of $n$ qubits uses $(L+1) \cdot n$ parameters, one per qubit and
    rotation layer.

    Because $R_Y$ and CX both have real matrix representations, the ansatz maps
    real amplitudes to real amplitudes. From $\ket{0}^{\otimes n}$, or after an
    encoding with real amplitudes such as angle encoding about the Y axis, it
    explores only the real subspace of the Hilbert space, which contains a state
    for every value an expectation of a real observable can take.

    After an encoding that introduces complex phases, such as dense angle
    encoding or angle encoding about the X or Z axis, the ansatz rotates the
    real and imaginary parts of the state separately and never mixes them. The
    expectation value of a real observable is then a sum of one contribution
    from each part, which restricts the functions of the features the model can
    represent. An ansatz with rotations about further axes lifts this
    restriction.

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
    Returns the number of parameters of a :meth:`Ansatz.real_amplitudes` ansatz.

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
        from qrisp.qml import real_amplitudes_size

        w = np.random.uniform(0, 2 * np.pi, real_amplitudes_size(3, layers=2))

    """
    return (layers + 1) * num_qubits


def _data_reuploading(qv, x, w, ansatz_size, encoding, ansatz, layers=2):
    r"""
    Applies alternating encoding and ansatz layers to a QuantumVariable.

    The feature vector is encoded once per layer rather than only at the start
    of the circuit:

    .. math::

        W(w_L)\, S(x) \cdots W(w_1)\, S(x)

    where $S$ is the encoding and $W$ the parameterized ansatz, acting from
    right to left.

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
    encoding : callable
        Applied as ``encoding(qv, x)`` at the start of each layer.
    ansatz : callable
        Applied as ``ansatz(qv, w_layer)`` after each encoding.

    Raises
    ------
    ValueError
        If the number of parameters does not match the layers.

    """
    if np.size(w) != layers * ansatz_size:
        raise ValueError(
            f"Tried to apply data reuploading with {np.size(w)} parameters over "
            f"{layers} layers. Expected {layers * ansatz_size} parameters, "
            f"{ansatz_size} for each layer."
        )

    w = _flatten_parameters(w)

    for layer in range(layers):
        encoding.apply(qv, x)
        ansatz.apply(qv, w[layer * ansatz_size : (layer + 1) * ansatz_size])

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
    Returns the number of parameters of a :meth:`Ansatz.hardware_efficient` ansatz.

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
        from qrisp.qml import hardware_efficient_size

        w = np.random.uniform(0, 2 * np.pi, hardware_efficient_size(3, layers=2))

    Raises
    ------
    ValueError
        If an axis follows itself in ``rotations``.

    """
    for axis, next_axis in zip(rotations, rotations[1:]):
        if axis == next_axis:
            raise ValueError(
                f"Repeated rotation axis in {tuple(rotations)} for the hardware efficient ansatz. "
                f"Consecutive rotations about the same axis compose into a single one, "
                f"so the repetition adds parameters the circuit cannot use. An axis may "
                f"recur once another axis separates the repeats, as in ('z', 'y', 'z')."
            )

    return (layers + 1) * len(rotations) * num_qubits


class Ansatz:
    r"""
    A configured parameterized circuit.

    An ansatz is built once, for a given register size, and applied once per
    evaluation. The classmethods :meth:`hardware_efficient`,
    :meth:`real_amplitudes` and :meth:`data_reuploading` each return one:

    ::

        from qrisp import QuantumFloat
        from qrisp.jasp import jaspify
        from qrisp.qml import Ansatz

        ansatz = Ansatz.real_amplitudes(num_qubits=3, layers=2)

        @jaspify
        def circuit(w):
            qf = QuantumFloat(3)
            ansatz.apply(qf, w)
            return qf

    **Why the qubit count is configured up front**

    The number of parameters an ansatz expects is a function of the register
    size. Under :ref:`Jasp <jasp>` tracing that size is a tracer rather than a
    number, so the expected count cannot be computed there and a wrong number of
    parameters is accepted silently: the surplus is ignored, or a short vector is
    read past its end by reusing its last value.

    Recording ``num_qubits`` while it is still an ordinary integer closes that
    gap. The expected count becomes a plain number, the parameter vector's size
    is available as a static shape, and the comparison therefore happens at trace
    time, before any gate is emitted.

    **Register width is the caller's responsibility**

    As for :ref:`Encoding <Encoding>`, ``num_qubits`` is not checked against the
    register the ansatz is applied to, which remains a tracer. Applying an ansatz
    configured for three qubits to a register of four is not reported. Keeping
    the two in step is left to the caller.

    **Writing an ansatz**

    Subclassing this class gives an ansatz of your own the same parameter count
    check and error messages as the built-in ones. Implement
    :meth:`apply_gates` to emit the circuit, and :meth:`parameter_count` so that
    the number of parameters follows from the register size rather than being
    restated at every construction:

    ::

        from jax.tree_util import register_pytree_node_class
        from qrisp import ry
        from qrisp.jasp import jrange
        from qrisp.qml import Ansatz

        @register_pytree_node_class
        class RotationAnsatz(Ansatz):
            def parameter_count(self, num_qubits):
                return num_qubits

            def apply_gates(self, qv, w):
                for i in jrange(qv.size):
                    ry(w[i], qv[i])
                return qv

        ansatz = RotationAnsatz(num_qubits=3)

    ``num_parameters`` is then derived on construction, and ``apply`` rejects a
    parameter vector that does not match it. Registering the class as a PyTree
    lets it be passed as an argument to a traced function rather than only
    captured from the enclosing scope. Settings of your own are kept across that
    roundtrip by returning them from ``_options``.

    """

    def __init__(self, num_qubits, num_parameters=None):
        if not isinstance(num_qubits, (int, np.integer)) or isinstance(num_qubits, bool):
            raise ValueError(f"Tried to configure an ansatz for {num_qubits!r} qubits, which is not an integer.")

        if num_qubits < 1:
            raise ValueError(f"Tried to configure an ansatz for {num_qubits} qubits.")

        self.num_qubits = int(num_qubits)

        if num_parameters is None:
            num_parameters = self.parameter_count(self.num_qubits)

        self.num_parameters = int(num_parameters)

    @classmethod
    def hardware_efficient(cls, num_qubits, layers=1, rotations=("y",), entangler="cx"):
        r"""
        Returns an ansatz of rotation layers separated by entangling layers.

        Each rotation layer applies one parameterized rotation per qubit and
        rotation axis, and each entangling layer a two qubit gate to
        neighbouring qubits. There is one more rotation layer than entangling
        layers, so the circuit both begins and ends with rotations.

        Parameters
        ----------
        num_qubits : int
            The size of the QuantumVariable the ansatz is applied to.
        layers : int, optional
            The number of rotation and entangling layer repetitions. The default
            is ``1``.
        rotations : tuple[str], optional
            The rotation axes applied to each qubit in a rotation layer, in
            order. The default is ``("y",)``. An axis may recur when another
            axis separates the repeats: ``("z", "y", "z")`` parameterizes an
            arbitrary single qubit rotation up to a global phase.
        entangler : {'cx', 'cz'}, optional
            The two qubit gate of the entangling layers. The default is
            ``'cx'``.

        Returns
        -------
        Ansatz
            The configured ansatz.

        Examples
        --------

        ::

            import numpy as np
            from qrisp import QuantumFloat
            from qrisp.qml import Ansatz

            qf = QuantumFloat(3)
            ansatz = Ansatz.hardware_efficient(num_qubits=3, layers=2)
            w = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)
            ansatz.apply(qf, w)

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

        _check_layers(layers, "hardware efficient")

        ansatz = _HardwareEfficient(num_qubits, hardware_efficient_size(num_qubits, layers, rotations))
        ansatz.layers = layers
        ansatz.rotations = rotations
        ansatz.entangler = entangler

        return ansatz

    @classmethod
    def real_amplitudes(cls, num_qubits, layers=1, entanglement="ring"):
        r"""
        Returns an ansatz preparing states with real amplitudes.

        Each rotation layer applies one $R_Y$ rotation per qubit, separated by
        entangling layers of CX gates. As both gates have real matrix entries,
        a state with real amplitudes, such as the initial state of a register,
        keeps real amplitudes.

        Parameters
        ----------
        num_qubits : int
            The size of the QuantumVariable the ansatz is applied to.
        layers : int, optional
            The number of rotation and entangling layer repetitions. The default
            is ``1``.
        entanglement : {'ring', 'linear'}, optional
            The shape of the entangling layers. The default is ``'ring'``, which
            closes the chain by entangling the last qubit with the first.

        Returns
        -------
        Ansatz
            The configured ansatz.

        Examples
        --------

        ::

            import numpy as np
            from qrisp import QuantumFloat
            from qrisp.qml import Ansatz

            qf = QuantumFloat(3)
            ansatz = Ansatz.real_amplitudes(num_qubits=3, layers=2)
            w = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)
            ansatz.apply(qf, w)

        Every amplitude of the resulting state is real.

        >>> np.allclose(np.array(qf.qs.statevector("array")).imag, 0)
        True

        """
        if entanglement not in _ENTANGLEMENTS:
            raise ValueError(
                f"Unknown entanglement '{entanglement}' for the real amplitudes "
                f"ansatz. Supported entanglements are {sorted(_ENTANGLEMENTS)}."
            )

        _check_layers(layers, "real amplitudes")

        ansatz = _RealAmplitudes(num_qubits, real_amplitudes_size(num_qubits, layers))
        ansatz.layers = layers
        ansatz.entanglement = entanglement

        return ansatz

    @classmethod
    def data_reuploading(cls, num_qubits, encoding, ansatz, layers=2):
        r"""
        Returns an ansatz alternating encoding and ansatz layers.

        The feature vector is encoded once per layer rather than only at the
        start of the circuit:

        .. math::

            W(w_L)\, S(x) \cdots W(w_1)\, S(x)

        where $S$ is the encoding and $W$ the parameterized ansatz, acting from
        right to left.

        Re-encoding the same features between parameterized layers raises the
        frequency spectrum the circuit can represent. A single encoding layer
        followed by an ansatz produces a function of the features whose Fourier
        spectrum is limited by the encoding alone, so no amount of added
        parameters widens it. With $L$ encoding layers the accessible spectrum
        grows with $L$, which is what allows a circuit on few qubits to fit a
        rich function.

        Unlike the other ansatzes this one also takes the features, so it is
        applied as ``apply(qv, w, x)``.

        Parameters
        ----------
        num_qubits : int
            The size of the QuantumVariable the ansatz is applied to.
        encoding : Encoding
            Applied at the start of each layer.
        ansatz : Ansatz
            Applied after each encoding.
        layers : int, optional
            The number of encoding and ansatz layer repetitions. The default is
            ``2``, the smallest number that re-uploads the features at all.

        Returns
        -------
        Ansatz
            The configured ansatz.

        Examples
        --------

        ::

            import numpy as np
            from qrisp import QuantumFloat
            from qrisp.qml import Ansatz, Encoding

            qf = QuantumFloat(2)
            reuploading = Ansatz.data_reuploading(
                num_qubits=2,
                encoding=Encoding.angle(n_features=2),
                ansatz=Ansatz.real_amplitudes(num_qubits=2),
                layers=3,
            )
            w = np.random.uniform(0, 2 * np.pi, reuploading.num_parameters)
            reuploading.apply(qf, w, [0.4, 1.2])

        """
        _check_layers(layers, "data reuploading")

        # A reuploading ansatz takes the features as a third argument, while the
        # inner ansatz is applied as apply(qv, w_layer). Nesting one inside
        # another would therefore raise a TypeError once applied, so it is
        # rejected here rather than at the first evaluation.
        if isinstance(ansatz, _DataReuploading):
            raise ValueError(
                "Tried to use a data reuploading ansatz as the inner ansatz of "
                "another one. The inner ansatz is applied without the features, "
                "which a reuploading ansatz requires. Pass an ansatz such as "
                "Ansatz.real_amplitudes, and raise layers to re-upload more often."
            )

        if ansatz.num_qubits != num_qubits:
            raise ValueError(
                f"Tried to configure a data reuploading ansatz for {num_qubits} qubits "
                f"around an inner ansatz configured for {ansatz.num_qubits}. Both act "
                "on the same register, so they must be configured for the same size."
            )

        reuploading = _DataReuploading(num_qubits, layers * ansatz.num_parameters)
        reuploading.layers = layers
        reuploading.encoding = encoding
        reuploading.ansatz = ansatz

        return reuploading

    def apply(self, qv, w):
        """
        Applies the ansatz to ``qv``.

        Parameters
        ----------
        qv : QuantumVariable
            The QuantumVariable to apply the ansatz to. Its size is not
            validated; see the note on register width in the class docstring.
        w : list[float] or numpy.ndarray
            The parameters, either flat or shaped.

        Returns
        -------
        QuantumVariable
            The QuantumVariable the ansatz was applied to.

        Raises
        ------
        ValueError
            If ``w`` does not hold exactly ``num_parameters`` parameters.

        """
        self._check_parameters(w)

        return self.apply_gates(qv, w)

    def apply_gates(self, qv, w):
        """
        Emits the gates of the ansatz.

        Subclasses implement this to define their circuit. It is called by
        :meth:`apply`, which has already rejected a parameter vector of the
        wrong size, so an implementation can rely on ``w`` holding exactly
        ``num_parameters`` parameters.

        Parameters
        ----------
        qv : QuantumVariable
            The QuantumVariable to apply the gates to. Under tracing its size
            is a tracer, so it cannot be compared against ``num_qubits`` or
            used to size an allocation. Loop over it with
            :ref:`jrange <jrange>` rather than ``range(len(qv))``.
        w : list[float] or numpy.ndarray
            The parameters, either flat or shaped.

        Returns
        -------
        QuantumVariable
            The QuantumVariable the gates were applied to.

        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement apply_gates, which an ansatz needs in order to emit its gates."
        )

    def parameter_count(self, num_qubits):
        """
        Returns the number of parameters the ansatz expects.

        Subclasses override this so that the count follows from the register
        size rather than being restated by the caller. It is called once,
        during construction, and the result is available as
        ``num_parameters``.

        Parameters
        ----------
        num_qubits : int
            The size the ansatz is configured for, as an ordinary integer.

        Returns
        -------
        int
            The number of parameters.

        """
        raise NotImplementedError(
            f"{type(self).__name__} neither implements parameter_count nor was "
            f"given a num_parameters, so the number of parameters it expects is unknown."
        )

    def _check_parameters(self, w):
        """
        Rejects a parameter vector of the wrong size.

        np.size reads a static shape, and num_parameters was computed outside
        the traced region, so unlike a count derived from qv.size this
        comparison also runs while tracing.
        """
        if np.size(w) != self.num_parameters:
            raise ValueError(
                f"Tried to apply an ansatz configured for {self.num_qubits} qubits "
                f"with {np.size(w)} parameters. Expected {self.num_parameters}."
            )

    def tree_flatten(self):
        """
        PyTree flatten for JAX.

        The configuration is auxiliary data rather than a child, so that
        ``num_qubits`` and ``num_parameters`` stay plain integers that the
        trace time check can compare against.
        """
        return (), (self.num_qubits, self.num_parameters, self._options())

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        PyTree unflatten for JAX.
        """
        num_qubits, num_parameters, options = aux_data
        ansatz = cls(num_qubits, num_parameters)

        for name, value in options.items():
            setattr(ansatz, name, value)

        return ansatz

    def __setstate__(self, state):
        # A subclass defined in a script is rebuilt from its code when loaded
        # in another process, without running the decorator that registered it
        # as a PyTree. Unregistered, it is its own single leaf. treedef_is_leaf
        # cannot tell, as it also holds for a registered node without children.
        self.__dict__.update(state)
        leaves = tree_leaves(self)

        if len(leaves) == 1 and leaves[0] is self:
            register_pytree_node_class(type(self))

    def _options(self):
        """
        Returns the per-ansatz settings, which subclasses set after building.
        """
        return {}


def _check_layers(layers, name):
    """
    Rejects a layer count that is not a non-negative integer.

    The parameter count check does not catch a negative count, which expects no
    parameters and would leave the QuantumVariable untouched. A fractional
    count would yield a fractional parameter count, truncated on construction.
    """
    if not isinstance(layers, (int, np.integer)) or isinstance(layers, bool):
        raise ValueError(f"Tried to configure a {name} ansatz with {layers!r} layers, which is not an integer.")

    if layers < 0:
        raise ValueError(f"Tried to configure a {name} ansatz with a negative layer count ({layers}).")


@register_pytree_node_class
class _HardwareEfficient(Ansatz):
    def _options(self):
        return {"layers": self.layers, "rotations": self.rotations, "entangler": self.entangler}

    def apply_gates(self, qv, w):
        return _hardware_efficient(qv, w, layers=self.layers, rotations=self.rotations, entangler=self.entangler)


@register_pytree_node_class
class _RealAmplitudes(Ansatz):
    def _options(self):
        return {"layers": self.layers, "entanglement": self.entanglement}

    def apply_gates(self, qv, w):
        return _real_amplitudes(qv, w, layers=self.layers, entanglement=self.entanglement)


@register_pytree_node_class
class _DataReuploading(Ansatz):
    def _options(self):
        return {"layers": self.layers, "encoding": self.encoding, "ansatz": self.ansatz}

    def apply(self, qv, w, x):
        """
        Applies alternating encoding and ansatz layers to ``qv``.

        Parameters
        ----------
        qv : QuantumVariable
            The QuantumVariable to apply the circuit to.
        w : list[float] or numpy.ndarray
            The parameters of all ansatz layers, concatenated.
        x : list[float] or numpy.ndarray
            The feature vector, encoded once per layer.

        Returns
        -------
        QuantumVariable
            The QuantumVariable the circuit was applied to.

        Raises
        ------
        ValueError
            If ``w`` does not hold exactly ``num_parameters`` parameters.

        """
        self._check_parameters(w)

        return _data_reuploading(
            qv,
            x,
            w,
            self.ansatz.num_parameters,
            self.encoding,
            self.ansatz,
            layers=self.layers,
        )

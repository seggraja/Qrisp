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

__all__ = ["expectation"]


def expectation(state_prep, observables, precision=0.01):
    r"""
    Estimates the expectation values of one or more observables.

    A single observable yields a scalar, a sequence of observables yields one
    entry per observable:

    .. math::

        \left( \braket{O_0}, \ldots, \braket{O_{m-1}} \right)

    Each observable is estimated from its own sampling run, so the entries stay
    separate. This is what a model with several outputs requires: an operator
    such as ``Z(0) + Z(1)`` combines the qubits into a single number, from which
    the individual contributions can no longer be recovered.

    The state preparation is sampled once per observable, so the cost grows
    with the length of ``observables``. Passing a single operator that already
    expresses the quantity of interest is therefore cheaper than combining
    several entries afterwards.

    Parameters
    ----------
    state_prep : callable
        A function returning a QuantumVariable. The expectation values are
        estimated for the state this function prepares. It can only take
        classical values as arguments, since a quantum value would have to be
        copied for each sampling iteration.
    observables : QubitOperator or iterable[QubitOperator]
        The observable, or a sequence of observables to estimate separately.
        Any iterable is accepted, including a numpy array of operators.
    precision : float, optional
        The precision with which each expectation value is estimated. The
        number of shots scales quadratically with the inverse precision. The
        default is ``0.01``.

    Returns
    -------
    callable
        A function that takes the arguments of ``state_prep`` and returns the
        expectation value, or an array of one expectation value per observable.

    Raises
    ------
    ValueError
        If ``observables`` is an empty sequence.

    Examples
    --------

    We estimate a single observable for an encoded feature vector.

    ::

        import numpy as np
        from qrisp import QuantumFloat
        from qrisp.operators import Z
        from qrisp.qml import Encoding, expectation

        encoding = Encoding.angle(n_features=2)

        def state_prep(x):
            qf = QuantumFloat(2)
            encoding.apply(qf, x)
            return qf

        ev = expectation(state_prep, Z(0))

    The features are all zero, so the register stays in $\ket{00}$ and the
    first qubit is measured as $+1$.

    >>> ev([0, 0])
    1.0

    Passing a sequence returns one entry per observable, which is how a model
    with one output per class is read out.

    ::

        ev = expectation(state_prep, [Z(0), Z(1)])

    >>> ev([np.pi, 0])
    array([-1.,  1.])

    **Hybrid models**

    The returned function takes and returns Jax arrays, so it composes with
    classical layers written in ``jnp`` on either side of it:

    ::

        quantum_block = expectation(state_prep, [Z(0), Z(1)])

        def model(x, p):
            h = jnp.tanh(x * p["w_in"])                     # classical
            q = quantum_block(h, p["theta"])                # quantum
            return jnp.tanh(jnp.dot(q, p["w_out"]))         # classical

    The optimizers take a single flat vector, while a model like this has
    named groups of parameters. ``jax.flatten_util.ravel_pytree`` converts
    between the two, and the unravel function it returns can be called inside
    the traced region, so the whole model trains through one flat vector:

    ::

        from jax.flatten_util import ravel_pytree
        from qrisp.jasp import jaspify, minimize

        parameters = {"w_in": jnp.ones(2), "theta": jnp.zeros(4),
                      "w_out": jnp.ones(2)}
        flat, unravel = ravel_pytree(parameters)

        @jaspify
        def train(flat_parameters):
            def loss(v):
                p = unravel(v)
                return (model(x, p) - y) ** 2

            return minimize(loss, flat_parameters, method="COBYLA").x

    Slicing the flat vector by hand instead is a common source of silent
    error: an off-by-one hands the wrong numbers to the wrong layer, which
    trains without raising anything.

    Note the optimizers available through :func:`qrisp.jasp.minimize` are
    gradient-free, so a hybrid model is trained without differentiating
    through the quantum block.

    """
    # A single operator is identified by what it provides rather than by what
    # it is, so that any iterable of observables is read as a sequence.
    if hasattr(observables, "expectation_value"):
        return observables.expectation_value(state_prep, precision=precision)

    observables = list(observables)

    if not observables:
        raise ValueError("Tried to estimate the expectation values of an empty sequence of observables.")

    ev_functions = [observable.expectation_value(state_prep, precision=precision) for observable in observables]

    def expectation_eval_function(*args):
        return jnp.array([ev_function(*args) for ev_function in ev_functions])

    return expectation_eval_function

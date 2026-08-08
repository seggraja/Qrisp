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
    observables : QubitOperator or list[QubitOperator]
        The observable, or a sequence of observables to estimate separately.
    precision : float, optional
        The precision with which each expectation value is estimated. The
        number of shots scales quadratically with the inverse precision. The
        default is ``0.01``.

    Returns
    -------
    callable
        A function that takes the arguments of ``state_prep`` and returns the
        expectation value, or an array of one expectation value per observable.

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

    """
    if not isinstance(observables, (list, tuple)):
        return observables.expectation_value(state_prep, precision=precision)

    ev_functions = [observable.expectation_value(state_prep, precision=precision) for observable in observables]

    def expectation_eval_function(*args):
        return jnp.array([ev_function(*args) for ev_function in ev_functions])

    return expectation_eval_function

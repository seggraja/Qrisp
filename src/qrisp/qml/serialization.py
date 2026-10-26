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

from importlib.metadata import PackageNotFoundError, version

import dill

__all__ = ["load_model", "save_model"]

# Increase whenever the layout of the saved record changes.
_FORMAT_VERSION = 1


def _qrisp_version():
    try:
        return version("qrisp")
    except PackageNotFoundError:
        return None


def save_model(filename, **parts):
    r"""
    Saves the parts of a model to a file.

    Each part is passed as a keyword argument and stored under that name, so
    that :func:`load_model` returns it under the same name. Parts are stored
    with their configuration, such as the scaling range of an encoding.
    Classes you define yourself, such as an :class:`Encoding` subclass written
    in a notebook, are stored together with their code.

    Parameters
    ----------
    filename : str or os.PathLike
        The file to write. An existing file is overwritten.
    **parts
        The parts of the model. Any object that can be pickled is accepted,
        including encodings, ansatzes, operators, and NumPy or Jax arrays.

    Raises
    ------
    ValueError
        If no parts are given.

    Examples
    --------

    ::

        import numpy as np
        from qrisp.operators import Z
        from qrisp.qml import Ansatz, Encoding, save_model

        encoding = Encoding.angle(n_features=2, data_range=([3.0, 1.0], [6.9, 2.5]))
        ansatz = Ansatz.real_amplitudes(num_qubits=2, layers=1)
        parameters = np.array([0.4, 1.1, 2.0, 0.7])

        save_model("model.pkl", encoding=encoding, ansatz=ansatz, observable=Z(0), parameters=parameters)

    """
    if not parts:
        raise ValueError(
            "Tried to save a model with no parts. Pass each part as a keyword argument, for example parameters=w."
        )

    record = {"format_version": _FORMAT_VERSION, "qrisp_version": _qrisp_version(), "parts": parts}

    # Without recurse, a class defined in a notebook is saved without the
    # functions its methods call, and fails when used after loading.
    with open(filename, "wb") as file:
        dill.dump(record, file, recurse=True)


def load_model(filename):
    r"""
    Loads the parts of a model saved with :func:`save_model`.

    .. warning::

        Loading a file runs code stored in it, as with any pickle-based format.
        Only load files from a source you trust.

    Parameters
    ----------
    filename : str or os.PathLike
        The file written by :func:`save_model`.

    Returns
    -------
    dict
        The saved parts, by the names they were saved under.

    Raises
    ------
    ValueError
        If the file was not written by :func:`save_model`, or uses a newer
        file format than this version of Qrisp reads.

    Examples
    --------

    We save a small model, then load it as we would in a new session. The parts
    come back by name. The encoding still holds its scaling range, so it takes
    measurements in their original units.

    ::

        import numpy as np
        from qrisp import QuantumFloat
        from qrisp.operators import Z
        from qrisp.qml import Ansatz, Encoding, expectation, load_model, save_model

        save_model(
            "model.pkl",
            encoding=Encoding.angle(n_features=2, data_range=([3.0, 1.0], [6.9, 2.5])),
            ansatz=Ansatz.real_amplitudes(num_qubits=2, layers=1),
            observable=Z(0),
            parameters=np.array([0.4, 1.1, 2.0, 0.7]),
        )

        model = load_model("model.pkl")

        def prepare():
            qf = QuantumFloat(2)
            model["encoding"].apply(qf, [5.1, 1.6])
            model["ansatz"].apply(qf, model["parameters"])
            return qf

        prediction = expectation(prepare, model["observable"])()

    """
    with open(filename, "rb") as file:
        record = dill.load(file)

    if not isinstance(record, dict) or "format_version" not in record:
        raise ValueError(f"Tried to load {filename}, which was not written by save_model.")

    if record["format_version"] > _FORMAT_VERSION:
        raise ValueError(
            f"Tried to load {filename}, which uses model file format {record['format_version']} (written by Qrisp "
            f"{record['qrisp_version']}). This version of Qrisp reads up to format {_FORMAT_VERSION}; update Qrisp to "
            "load it."
        )

    if not isinstance(record.get("parts"), dict):
        raise ValueError(f"Tried to load {filename}, which was not written by save_model.")

    return dict(record["parts"])

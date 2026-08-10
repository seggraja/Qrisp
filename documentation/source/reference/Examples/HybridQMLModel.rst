.. _HybridQMLModel:

Hybrid quantum-classical model
==============================

This example shows how to place trainable classical layers on both sides of a
quantum block, and how to hand the resulting parameters to an optimizer.

::

   x  ->  classical  ->  quantum block  ->  classical  ->  prediction
          (jnp ops)      (encoding +        (jnp ops)
                          ansatz)

:meth:`expectation <qrisp.qml.expectation>` returns an ordinary function of
arrays, so the quantum part composes with ``jnp`` operations like any other
step. Ask for one observable per output you want to feed forward: a list of
observables gives a vector rather than a single number.

The model
---------

::

   import numpy as np
   import jax.numpy as jnp

   from qrisp import QuantumFloat
   from qrisp.operators import Z
   from qrisp.qml import Ansatz, Encoding, expectation
   from qrisp.jasp import jaspify, minimize

   num_qubits = 2

   encoding = Encoding.angle(n_features=num_qubits)
   ansatz = Ansatz.real_amplitudes(num_qubits=num_qubits, layers=1)


   def circuit(x, w):
       qf = QuantumFloat(num_qubits)
       encoding.apply(qf, x)
       ansatz.apply(qf, w)
       return qf


   # Two observables, so the readout is a length-2 vector.
   quantum_block = expectation(circuit, [Z(0), Z(1)], precision=0.05)


   def hybrid_model(x, p):
       h = jnp.tanh(x * p["w_in"] + p["b_in"])                 # classical
       q = quantum_block(h, p["theta"])                        # quantum
       return jnp.tanh(jnp.dot(q, p["w_out"]) + p["b_out"])    # classical

Named parameters and the flat vector
------------------------------------

:meth:`minimize <qrisp.jasp.minimize>` takes a single flat vector, while the
model above wants a dictionary. Converting between the two by hand, slicing the
vector with computed offsets, is a common source of silent error: an off-by-one
hands the wrong numbers to the wrong layer and nothing is raised, so the model
simply trains on the wrong values.

``jax.flatten_util.ravel_pytree`` does the conversion. It flattens any nested
structure of arrays into one vector and returns an ``unravel`` function that
restores the exact shapes, including scalars. That ``unravel`` can be called
inside the traced function, which is where the optimizer needs it.

::

   from jax.flatten_util import ravel_pytree

   parameters = {
       "w_in": jnp.ones(num_qubits),
       "b_in": jnp.zeros(num_qubits),
       "theta": jnp.full(ansatz.num_parameters, 0.5),
       "w_out": jnp.ones(num_qubits),
       "b_out": jnp.array(0.0),
   }

   flat0, unravel = ravel_pytree(parameters)

>>> flat0.shape
(11,)
>>> sorted(unravel(flat0))
['b_in', 'b_out', 'theta', 'w_in', 'w_out']

Training
--------

The loss unravels the vector it is given and then calls the model normally, so
the optimizer never sees the structure and the model never sees the flat
vector. The first feature decides the label here, which keeps the task within
what this circuit can represent.

::

   X_train = jnp.array([[0.0, 0.0], [0.0, np.pi], [np.pi, 0.0], [np.pi, np.pi]])
   y_train = jnp.array([1.0, 1.0, -1.0, -1.0])


   def loss(v):
       p = unravel(v)
       total = 0.0
       for i in range(X_train.shape[0]):
           total = total + (hybrid_model(X_train[i], p) - y_train[i]) ** 2
       return total / X_train.shape[0]


   @jaspify(terminal_sampling=True)
   def train():
       results = []
       for scale in [0.5, 1.5]:
           v0 = flat0 * scale
           result = minimize(loss, v0, method="COBYLA", options={"maxiter": 60})
           results.append((loss(v0), result.fun, result.x))
       return results


   results = train()

The solvers behind ``minimize`` are local, so several starting points are
trained and the best one is kept.

::

   for before, after, _ in results:
       print(f"loss {float(before):.3f} -> {float(after):.3f}")

   best = min(results, key=lambda r: float(r[1]))

The expectation values are estimated from samples, so the figures move from run
to run. Both starting points drop from around 0.8 to roughly 0.3:

::

   loss 0.829 -> 0.322
   loss 1.103 -> 0.294

Reading the trained parameters back out is the same call in reverse, and both
halves of the model have moved.

::

   p_opt = unravel(jnp.asarray(best[2]))

>>> sorted(p_opt)
['b_in', 'b_out', 'theta', 'w_in', 'w_out']
>>> bool(np.allclose(p_opt["theta"], parameters["theta"]))
False
>>> bool(np.allclose(p_opt["w_out"], parameters["w_out"]))
False

What this does and does not give you
------------------------------------

The composition is real: classical layers train alongside the quantum ones,
through one optimizer call, and the same code runs traced.

It is not end-to-end differentiation. ``minimize`` is gradient-free, so the
classical layers are optimized by sampling the loss rather than by
backpropagation. That is workable for the small parameter counts here, but it
does not scale the way a classical network does, which is why the classical
layers in this example are kept small.

Note also the cost of a wide readout: each expectation value is estimated from
its own sampling run, so ``m`` observables cost ``m`` runs even when the
observables commute and one run would suffice.

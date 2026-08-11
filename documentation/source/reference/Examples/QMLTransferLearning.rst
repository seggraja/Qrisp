.. _QMLTransferLearning:

Transfer learning from a pretrained network
==========================================

A pretrained image network already knows how to turn a photograph into a
compact list of numbers. This example keeps that network **frozen** and trains
only a small quantum model on the features it produces, classifying cats
against dogs with seven parameters.

::

   photo  ->  frozen ResNet-18  ->  512 features  ->  PCA  ->  quantum head
              (not trained)                          (2 dims)  (7 parameters)

Freezing the backbone is what makes this practical. A frozen network has no
trainable parameters, so its output is a constant for a given image. The
features can therefore be computed **once, outside any traced function**, and
from that point on the quantum model sees an ordinary array.

It also sidesteps the missing piece. Training the backbone would need gradients
flowing back through the quantum block, and the optimizers reachable from
:meth:`minimize <qrisp.jasp.minimize>` are gradient-free. Freezing means there
is nothing to send a gradient into.

Requirements
------------

Qrisp itself needs nothing extra. The backbone and the dataset come from
``torchvision``, which is **not** a Qrisp dependency and is used here only to
obtain a pretrained network and some photographs:

.. list-table::
   :widths: 30 30
   :header-rows: 1

   * - Package
     - Version used here
   * - Python
     - 3.11.12
   * - ``torch``
     - 2.4.1
   * - ``torchvision``
     - 0.19.1
   * - ``numpy``
     - 2.4.6
   * - ``jax``
     - 0.7.1

::

   pip install torch==2.4.1 torchvision==0.19.1

On first run the script downloads the ResNet-18 weights (about 45 MB) and the
Oxford-IIIT Pet images (about 800 MB, which expand to roughly 1.6 GB on disk).
Any other source of features works equally well — a different framework, an
ONNX export, or an array saved earlier — because only the array reaches Qrisp.

The data
--------

Oxford-IIIT Pet ships a binary cat-or-dog label alongside its breed labels,
which is what ``target_types="binary-category"`` selects. Twenty-five images
per class are taken for training and another twenty-five, from the dataset's
own test split, are held back.

::

   import numpy as np
   import torch
   from torchvision.models import resnet18, ResNet18_Weights
   from torchvision.datasets import OxfordIIITPet

   import jax.numpy as jnp
   from jax.flatten_util import ravel_pytree

   from qrisp import QuantumFloat
   from qrisp.operators import Z
   from qrisp.qml import Ansatz, Encoding, expectation, minmax_scale
   from qrisp.jasp import jaspify, minimize

   weights = ResNet18_Weights.DEFAULT
   preprocess = weights.transforms()

   train_data = OxfordIIITPet(root=".", split="trainval",
                              target_types="binary-category", download=True)
   test_data = OxfordIIITPet(root=".", split="test",
                             target_types="binary-category", download=True)


   def take(dataset, per_class, seed):
       rng = np.random.default_rng(seed)
       labels = np.array(dataset._labels)
       chosen = np.concatenate([rng.permutation(np.where(labels == 0)[0])[:per_class],
                                rng.permutation(np.where(labels == 1)[0])[:per_class]])
       images = torch.stack([preprocess(dataset[i][0].convert("RGB")) for i in chosen])
       return images, np.array([1.0] * per_class + [-1.0] * per_class)


   per_class = 25
   train_images, y_train = take(train_data, per_class, seed=0)
   test_images, y_test = take(test_data, per_class, seed=1)

Cats are labelled ``+1`` and dogs ``-1``, matching the range of the
:math:`\langle Z \rangle` values the circuit returns.

Freezing the backbone
---------------------

Replacing the final classification layer with an identity makes the network
output its 512 penultimate features instead of class scores. Evaluation mode
and ``requires_grad_(False)`` together mean nothing in the backbone trains.

::

   backbone = resnet18(weights=weights)
   backbone.fc = torch.nn.Identity()
   backbone.eval()
   for parameter in backbone.parameters():
       parameter.requires_grad_(False)

   with torch.no_grad():
       train_features = backbone(train_images).numpy()
       test_features = backbone(test_images).numpy()

>>> train_features.shape
(50, 512)

Reducing to the qubit count
---------------------------

512 features do not fit in a small register, so they are projected onto the
directions along which the training features vary most. This is ordinary
classical work, done once. The projection is derived from the **training** set
only and then applied to both, so nothing about the test images influences it.

:meth:`minmax_scale <qrisp.qml.minmax_scale>` then maps the result into
rotation angles, again fitting on the training set and applying those bounds to
the test set.

::

   num_qubits = 2

   mean = train_features.mean(axis=0)
   _, _, principal = np.linalg.svd(train_features - mean, full_matrices=False)
   directions = principal[:num_qubits].T

   train_reduced = (train_features - mean) @ directions
   test_reduced = (test_features - mean) @ directions

   X_train, X_test = minmax_scale(train_reduced, test_reduced)
   X_train, X_test = jnp.array(X_train), jnp.array(X_test)

>>> X_train.shape
(50, 2)

Two components keep much less of the difference between cats and dogs than the
512 features held. That loss is real, and it is the main thing limiting the
result below.

The quantum head
----------------

From here nothing refers to the backbone. Two observables give a vector
readout, and a small classical layer turns it into one prediction.

::

   encoding = Encoding.angle(n_features=num_qubits)
   ansatz = Ansatz.real_amplitudes(num_qubits=num_qubits, layers=1)


   def circuit(x, w):
       qf = QuantumFloat(num_qubits)
       encoding.apply(qf, x)
       ansatz.apply(qf, w)
       return qf


   quantum_block = expectation(circuit, [Z(0), Z(1)], precision=0.15)


   def model(x, p):
       q = quantum_block(x, p["theta"])
       return jnp.tanh(jnp.dot(q, p["w_out"]) + p["b_out"])


   parameters = {
       "theta": jnp.full(ansatz.num_parameters, 0.5),
       "w_out": jnp.ones(num_qubits),
       "b_out": jnp.array(0.0),
   }

   flat0, unravel = ravel_pytree(parameters)

``ravel_pytree`` converts the named groups into the single flat vector the
optimizer expects, and back again. See :ref:`HybridQMLModel` for why that
conversion is worth leaving to Jax.

>>> int(flat0.size)
7

Seven parameters are trained in total. The backbone contributes none of them,
which is the point of freezing it.

Training
--------

::

   def make_loss(X, y):
       def loss(v):
           p = unravel(v)
           total = 0.0
           for i in range(X.shape[0]):
               total = total + (model(X[i], p) - y[i]) ** 2
           return total / X.shape[0]

       return loss


   @jaspify(terminal_sampling=True)
   def train():
       loss = make_loss(X_train, jnp.array(y_train))
       results = []
       for scale in [0.5, 1.5]:
           result = minimize(loss, flat0 * scale, method="COBYLA", options={"maxiter": 40})
           results.append((loss(flat0 * scale), result.fun, result.x))
       return results


   results = train()

   for before, after, _ in results:
       print(f"loss {float(before):.3f} -> {float(after):.3f}")

   best = min(results, key=lambda r: float(r[1]))

The solvers are local, so two starting points are tried and the better one is
kept. One loss evaluation samples the circuit once per image, so a run of forty
iterations takes several minutes. Expectation values are estimated from
samples, so the figures move from run to run:

::

   loss 0.480 -> 0.112
   loss 1.707 -> 0.030

Accuracy
--------

::

   @jaspify(terminal_sampling=True)
   def predictions(v, X):
       p = unravel(v)
       return jnp.array([model(X[i], p) for i in range(X.shape[0])])


   def accuracy(v, X, y):
       return float(np.mean(np.sign(np.asarray(predictions(v, X))) == y))

Two runs of the above reached ``1.0`` on the training set, and ``0.98`` and
``1.0`` on the held-out set — fifty unseen photographs, of which at most one
was misclassified. The held-out figure is the one worth quoting: the model is
separating photographs it never saw rather than memorising the training set.
Because the expectation values are sampled, the exact figure varies between
runs.

What this does and does not give you
------------------------------------

The classification is real, and it is the pretrained features that make seven
parameters enough. The backbone has already done the difficult part, turning
pixels into something nearly linearly separable; the quantum head only has to
draw a boundary in two dimensions.

That also bounds the result. Quality is limited by the frozen features and by
the reduction step — projecting 512 dimensions onto 2 discards a great deal,
and a poorer projection would lose the signal before the circuit ever saw it.
A comparable classical layer on the same two components would do about as well,
so this example demonstrates that the pattern works, not that the quantum head
is preferable.

The backbone is not fine-tuned, and cannot be with the optimizers available
here. End-to-end training of the whole stack needs gradients through the
quantum block, which is a separate piece of work.

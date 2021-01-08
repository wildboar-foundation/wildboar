=================
wildboar tutorial
=================

Machine learning
================
In general, the machine learning problem setting consists of n samples of data an the goal is to predict properties of unknown data.
``wildboar``, in particular consider machine learning problems in which the samples are data series, e.g., time series or
otherwise temporally or logically ordered data.

.. note::

    For solving general machine learning problems with Python, consider using `scikit-learn <https://scikit-learn.org/>`_


Similar to general machine learning problems, temporal machine learning consider problems that fall into different categories

* Supervised learning, in which the data series are labeled with additional information. The additional information can
  be either numerical or nominal

  * In classification problems the time series belong to one of two or more labels and the goal is to learn a function that can label unlabeled time series.

  * In regression problems the time series are labeled with a numerical attribute and the task is to assigned a new numerical value to an unlabeled time series.

Loading an example dataset
==========================
Wildboar bundles a few `standard datasets (no https) <http://www.timeseriesclassification.com/>`_ from the time series community.

In the example, we load the dataset ``synthetic_control`` and the ``TwoLeadECG`` dataset.

.. code-block:: python

    >>> from wildboar.datasets import load_synthetic_control, load_two_lead_ecg
    >>> x, y = load_synthetic_control()
    >>> x_train, x_test, y_train, y_test = load_two_lead_ecg(merge_train_test=False)

The datasets are Numpy ``ndarray`` with ``x.ndim==2`` and ``y.ndim==1``. We can get the number of samples and time points.

.. code-block:: python

    >>> n_samples, n_timestep = x.shape

.. note::

    By setting ``merge_train_test`` to `False`, the original training and testing splits from the UCR repository are preserved.

    A more robust and reliable method for splitting the datasets into training and testing partitions is to use the model selection functions from scikit-learn.

    .. code-block:: python

        >> from sklearn.model_selection import train_test_split
        >> x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

Learning and predicting
=======================
All estimators in wildboar implements the same interface as all estimators of scikit-learn. We can `fit` an estimator to an input dataset and `predict` the label of a new sample.

An example of a temporal estimator is the ``wildboar.ensemble.ShapeletForestClassifier`` which implements a random shapelet forest classifier.

.. code-block:: python

    >>> from wildboar.ensemble import ShapeletForestClassifier
    >>> clf = ShapeletForestClassifier()
    >>> clf.fit(x_train, y_train)

The classifier (``clf``) is fitted using the training samples, i.e., a model is inferred.

.. code-block:: python

    >>> clf.predict(x_test[-1:, :])
    array([6.])

.. note::

    The predict function expects an ``ndarray`` of shape ``(n_samples, n_timestep)``, where ``n_timestep`` is the size
    of training timestep.

Model persistence
=================

All `wildboar` models can be persisted to disk using `pickle <https://docs.python.org/3/library/pickle.html>`_

.. code-block:: python

    >>> import pickle
    >>> repr = pickle.dumps(clf) # clf fitted earlier
    >>> clf_ = pickle.loads(repr)
    >>> clf_.predict(x_test[-1:, :])
    array([6.])

.. note::

    Models persisted using an older versions of wildboar is not guaranteed to work when using a newer version (or vice versa).

.. warning::

    `The pickle module is not secure. Only unpickle data you trust. <https://docs.python.org/3/library/pickle.html>`_

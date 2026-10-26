.. currentmodule:: wildboar
.. _guide-datasets:

################
Interpretability
################


Time series classification involves categorizing sequences of data points,
typically collected over time, into predefined classes. While accurate
classification models are essential, understanding *why* a model makes certain
predictions is equally important. This is where interpretability and
explainability come into play.

**Interpretability** refers to the degree to which a human can understand the
cause of a decision made by a model. It involves providing insights into how
the input features (e.g., time points, trends, or patterns) influence the
model's output.

**Explainability** builds on interpretability by offering a more detailed and
often visual account of the decision-making process. It seeks to demystify
complex models, making them more transparent and trustworthy by showing, for
instance, which parts of the time series were most influential in determining
the final classification.

In time series classification, both interpretability and explainability are
crucial for ensuring that models are not just accurate, but also reliable and
understandable to users. This documentation provides an overview of the methods
and tools available to achieve these goals in time series classification tasks.

Wildboar offers multiple interpretability methods from global importance scores
to counterfactual explanations. Here is a minimal example of the API for both
counterfactual explainers and importance scores. First we need to fit a classifier:
here a weak :class:`~wildboar.linear_model.RandomShapeletClassifier`.

.. execute::
   :context:
   :show-return:

   from wildboar.datasets import load_gun_point
   from wildboar.linear_model import RandomShapeletClassifier

   X_train, X_test, y_train, y_test = load_gun_point(merge_train_test=False)
   f = RandomShapeletClassifier(n_shapelets=100, random_state=1)
   f.fit(X_train, y_train)
   f.score(X_test, y_test)

Next, we can compute the interval importance, that is the most influential
segments of the time series.

.. execute::
   :context:
   :show-return:

   from wildboar.explain import IntervalImportance
   explain = IntervalImportance(random_state=1)
   explain.fit(f, X_test[:10], y_test[:10])

   explain.importances_.mean

We can also plot the importances:

.. execute::
   :context:

   explain.plot(X_test[:10], k=3)

We can also compute counterfactual explanations, that is *how should we alter
the time series to change the class?*.

.. execute::
   :context:

   from wildboar.explain.counterfactual import NativeGuideCounterfactual

   cf = NativeGuideCounterfactual()
   cf.fit(f, X_train, y_train)
   x_cf = cf.explain(X_test[0:2], [2, 1])

   plt.plot(x_cf[1])
   plt.plot(X_test[1])

.. toctree::
   :maxdepth: 2
   :hidden:

   explain/counterfactuals

===================
Welcome to wildboar
===================

Wildboar is a package for supervised and unsupervised machine learning for time
series data.

.. toctree::
   :hidden:
   :maxdepth: 3

   install
   tutorial
   guide
   examples
   more
   

.. grid:: 2
    :gutter: 2 2 2 2

    .. grid-item-card:: Classification

        Identifying which category a time series belong to.

        **Algorithms**: Random shapelet forest, Extra shapelet trees and ROCKET.

        +++
        
        .. button-ref:: examples-supervised
           :color: secondary
           :expand:

           Examples


    .. grid-item-card:: Regression

        Predicting a continuous-valued attribute associated with a time series.

        **Algorithms**: Random shapelet forest, Extra shapelet trees and ROCKET.

        +++

        .. button-ref:: examples-supervised
           :color: secondary
           :expand:

           Examples   
   
    .. grid-item-card:: Explainability

        Explaining time series classifiers by counterfactual reasoning and by importance scores.

        **Algorithms**: Shapelet forest counterfactuals, KNearest counterfactuals and
        Prototype counterfactuals.

        +++

        .. button-ref:: examples-explainability
           :color: secondary
           :expand:

           Examples

    .. grid-item-card:: Unsupervised

         Identify interesting regions, motifs or outlier in time series. Or embed time series
         using various features.

         **Algorithms**: MatrixProfile, Isolation shapelet forests and Shapelet, ROCKET, and
         interval embeddings.

         +++
         
         .. button-ref:: examples-unsupervised
           :color: secondary
           :expand:

           Examples
   


.. grid:: 1

   .. grid-item::
      
      **News**
      
      .. include:: more/news.rst
         :end-before: ..end-first-page 
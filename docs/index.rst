.. div:: landing-title
   :style: padding: 0.1rem 0.5rem 0.6rem 0; background-image: linear-gradient(315deg, #973c8c 0%, #343067 74%); clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem)); -webkit-clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem));

   .. grid::
      :reverse:
      :gutter: 2 3 3 3
      :margin: 4 4 1 2

      .. grid-item::
         :columns: 12 4 4 4

         .. image:: ./_static/logo.svg
            :width: 200px
            :class: sd-m-auto dark-light wb-drop-shadow

      .. grid-item::
         :columns: 12 8 8 8
         :child-align: justify
         :class: sd-fs-3 sd-text-white sd-font-weight-bolder

         Efficient, simple and familiar temporal machine learning in Python.

         .. grid::
            :gutter: 0
            :margin: 0

            .. grid-item::
               :columns: 6

               .. button-ref:: install
                  :ref-type: doc
                  :outline:
                  :color: light
                  :class: sd-px-4 sd-fs-5

            .. grid-item::
               :columns: 6

               .. button-ref:: more/whatsnew
                  :ref-type: doc
                  :outline:
                  :color: light
                  :class: sd-px-4 sd-fs-5


.. grid:: 1 1 3 3

   .. grid-item-card::
      :padding: 2
      :link: guide
      :link-type: doc
      :text-align: center

      :fas:`person-running;10em` Getting started
      ^^^

      New to *Wildboar*? Check out the getting started guide! It contains an
      introduction to the main concepts.

   .. grid-item-card::
      :padding: 2
      :link: examples
      :link-type: doc
      :text-align: center

      :fas:`fire;10em` Examples
      ^^^

      The reference guide contains detailed descriptions of the Wildboar API.
      The reference describe what methods, classes and which parameters can be
      used.

   .. grid-item-card::
      :padding: 2
      :link: api/wildboar/index
      :link-type: doc
      :text-align: center

      :fas:`book;10em` API Reference
      ^^^

      The reference guide contains detailed descriptions of the Wildboar API.
      The reference describe what methods, classes and which parameters can be
      used.

.. toctree::
  :maxdepth: 3
  :hidden:

  Install <install>
  guide
  API <api/index>
  examples
  more/whatsnew


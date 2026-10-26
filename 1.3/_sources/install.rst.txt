################
Install wildboar
################

There are a few options to install wildboar:

- Install the latest official distribution from `PyPi
  <https://pypi.org/project/wildboar>`_. This is the recommended approach for
  most users.

- Build and compile the package from source. This provides the fastest binaries
  targeted for the specific platform.

Binary distributions are automatically built for macOS, GNU/Linux and Windows.
The binaries can be installed through `PyPi <https://pypi.org/project/wildboar>`_.

.. code-block:: shell

   pip install wildboar

If you are on a system where users don't have write-accesses to the location of
Python packages, the distribution can be installed in the user directory.

.. code-block:: shell

   pip install --user wildboar

You can also specify a specific version to install by replacing `wildboar` with,
e.g., `wildboar==1.2.0` where `1.2.0` is the version to install.

To avoid conflicts with already installed packages, it is strongly recommended
installing the package in a
`virtual environment <https://docs.python.org/3/tutorial/venv.html>`_. You can set
up a virtual environment using `venv`.

.. code-block:: shell

   python3 -m venv .venv # create a virtual environment in the folder .venv
   source .venv/bin/activate
   pip install wildboar

.. note::

   Depending on your operating system, there are some possible ceavets. While its
   outside the scope of this documentation to enumerate all of these, we have
   collected a few common issues here.

.. tab:: Debian

   For Debian based distributions `python3-venv` must be installed for virtual
   environments to work.

   .. code-block:: shell

      apt install python3-venv

.. tab:: MacOS

   For users of MacOS it is recommended to install python using
   `Homebrew <https://brew.sh/>`_

   .. code-block:: shell

      brew install python

.. tab:: Windows

   For users of Windows it is recommended to use
   `Anaconda <https://docs.conda.io/en/latest>`_ or
   `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. ``wildboar`` is
   still installed using ``pip``

.. toctree::
  :maxdepth: 3
  :hidden:

  install/build

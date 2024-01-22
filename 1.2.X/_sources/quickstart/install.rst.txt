################
Install wildboar
################

There are a few options to install wildboar:

- `Install the latest official distribution <#install-the-latest-release>`_ from
  `PyPi <https://pypi.org/project/wildboar>`_. This is the recommended approach
  for most users.

- `Build and compile the package from source <#build-and-compile-from-source>`_.
  This provides the fastest binaries targeted for the specific platform.

**************************
Install the latest release
**************************

Binary distributions are automatically built for macOS, GNU/Linux and Windows.
The binaries can be installed through [PyPi](https://pypi.org/project/wildboar).

.. code-block:: shell

   pip install wildboar

If you are on a system where users don't have write-accesses to the location of
Python packages, the distribution can be installed in the user directory.

.. code-block:: shell

   pip install --user wildboar

You can also specify a specific version to install by replacing `wildboar` with,
e.g., `wildboar==1.1.1` where `1.1.1` is the version to install.

.. warning::

   Due to a mistake in the distribution of version 1.0, these packages will try to install
   an incompatible version of ``numpy``. You can still install ``wildboar==1.0.12``, by manually
   installing ``numpy==1.19`` before installation.

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

*****************************
Build and compile from source
*****************************

Building from source is required to use the latest features or to work on new
features or pull requests. The process is simple.

1. Use `Git <https://git-scm.com/>`_ to fetch the latest version.

   .. code-block:: shell

      git clone --depth 10 https://github.com/isaksamsten/wildboar.git

2. Install a c-compiler for your operating system.

   .. tab:: Window

      First install `build tools for Visual Studio 2019
      <https://visualstudio.microsoft.com/downloads/>`_

      .. note::
         You *do not* need to install Visual Studio 2019. You only need the
         **Build Tools for Visual Studio 2019** which you can find under **All downloads** and then
         **Tools for Visual Studio 2019**.

      For 64-bit Python, configure the build environment by running the following commands in `cmd` console.

      .. code-block:: bat

         SET DISTUTILS_USE_SDK=1
         "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

      .. note::
         Replace the path with the install path of Visual Studio Build tools.

   .. tab:: macOS

      Install the macOS command line tools

      .. code-block:: shell

         xcode-select --install

   .. tab:: GNU/Linux

      Install build dependencies for your distribution.

      For Debian-based operating systems, e.g. Ubuntu:

      .. code-block:: shell

         apt install build-essential python3-dev python3-pip

3. Optionally create a new
   `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_.

4. Install the required Python-packages using pip

   .. code-block:: shell

      pip install -r requirements.txt

5. Build the project in editable mode to ease development

   .. code-block:: shell

      pip install --verbose --no-build-isolation --editable .

   Alternatively, if you just want to install Wildboar run:

   .. code-block:: shell

      pip install .

   .. note::
      The environment variable ``WILDBOAR_BUILD`` is used to control arguments to
      the build environment. Setting ``WILDBOAR_BUILD=optimized`` can build a
      version optimized for the current processor architecture.

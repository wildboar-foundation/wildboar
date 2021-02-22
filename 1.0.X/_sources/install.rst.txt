===================
Installing wildboar
===================

There are a few options to install wildboar:

* `Install the latest official distribution <#install-the-latest-release>`_ from `PyPi <https://pypi.org/project/wildboar/>`_. This is the recommended approach for most users.

* `Build and compile the package from source <#build-and-compile-from-source>`_. This provides the fastest binaries targeted for the specific platform.

Install the latest release
==========================
Binary distributions are built for MacOS, GNU/Linux and Windows.

.. code-block:: shell

    pip install wildboar

If you are on a system where users don't have write accesses to the location of Python packages, the distribution can be
installed in the user directory.

.. code-block:: shell

    pip install --user wildboar

To avoid conflicts with already installed packages, it is strongly recommended to install the package in a `virtual
environment <https://docs.python.org/3/tutorial/venv.html>`_.

.. code-block:: shell

    python3 -m venv .venv # create a virtual environment in the folder .venv
    source .venv/bin/activate
    pip install wildboar

.. note::

    For Debian based distributions ``python3-venv`` must be installed for virtual environments to work.

    .. code-block:: shell

        apt install python3-venv

.. note::

    For users of MacOS it is recommended to install python using `Homebrew <https://brew.sh/>`_

    .. code-block:: shell

        brew install python

.. note::

    For users of Windows it is recommended to use `Anaconda <https://docs.conda.io/en/latest/>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.
    wildboar is still installed using ``pip``

Build and compile from source
=============================

Building from source is required to use the latest features or to work on new features or pull requests.

1. Use `Git <https://git-scm.com/>`_ to checkout the latest version

    .. code-block:: shell

        git clone --depth 10 https://github.com/isaksamsten/wildboar.git

2) Install a c-compiler for `Windows <#windows>`_, `MacOS <#macos>`_ or `GNU/Linux <#gnu-linux>`_

3) Optionally create a new `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_.

4) Install the required Python-packages using pip

    .. code-block:: shell

        pip install -r requirements.txt

5) Build the project in editable mode to ease development

    .. code-block:: shell

        pip install --verbose --no-build-isolation --editable .

    .. note::

        The environment variable ``WILDBOAR_BUILD`` is used to control arguments to the build environment.
        Setting ``WILDBOAR_BUILD=optimized`` can build a version optimized for the current processor architecture.


Platform specific instructions
------------------------------

Windows
*******
First install `Build tools for Visual Studio 2019 <https://visualstudio.microsoft.com/downloads/>`_

.. note::

    You `do not` need to install Visual Studio 2019. You only need the **Build Tools for Visual Studio 2019**, under **All downloads** -> **Tools for Visual Studio 2019**.

For 64-bit Python, configure the build environment by running the following commands in ``cmd`` console.

.. code-block:: bat

    SET DISTUTILS_USE_SDK=1
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

.. note::

    Replace the path with the install path of Visual Studio Build tools.

MacOS
*****
Install the MacOS command line tools

.. code-block:: shell

    xcode-select --install

GNU/Linux
*********

Install build dependencies for Debian-based operating systems, e.g. Ubuntu:

.. code-block:: shell

    apt install build-essential python3-dev python3-pip

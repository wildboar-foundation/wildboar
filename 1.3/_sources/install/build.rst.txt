#################
Build from source
#################

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

3. Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_
   and enter the repository:

   .. code-block:: shell

      cd wildboar
      uv sync

   This creates ``.venv``, installs Wildboar in editable mode, and installs the
   development dependencies. On systems with Make, ``make deps`` runs the same
   command. Run it again when dependencies change.

Incremental development builds
==============================

After editing Cython code, rebuild extensions without reinstalling the package:

.. code-block:: shell

   make build

Run tests with:

.. code-block:: shell

   make test

To build the documentation, run ``make docs-deps`` followed by ``make docs``.
Run ``make docs-deps`` again after syncing if uv removed the documentation tools.

Build options
=============

The environment variable ``WILDBOAR_BUILD`` controls compiler options. Supported
values are ``default``, ``debug`` and ``optimize``. For example:

.. code-block:: shell

   WILDBOAR_BUILD=optimize make rebuild

Use ``make rebuild`` when changing compiler flags or build mode.

Additional ``build_ext`` options can be passed through Make, for example:

.. code-block:: shell

   make build BUILD_EXT_ARGS="--parallel 4"

Regular installation
====================

If you only want to install Wildboar from source rather than develop it, use:

.. code-block:: shell

   uv pip install .

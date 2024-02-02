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

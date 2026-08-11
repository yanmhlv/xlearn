Installation Guide for Windows
----------------------------------

For now, xLearn can support Windows. This page gives instructions on how to build and install the xLearn from source code on Windows. Before starting,  make sure that your Windows has already installed  ``Visual Studio`` (2022 or newer, for the C++23 support) and ``CMake``.

Install Visual Studio
^^^^^^^^^^^^^^^^^^^^^^^^

*If you have already installed your C++ compiler before, you can skip this step.*

Download Visual Studio ``vs_xxxx_xxxx.exe`` from https://visualstudio.microsoft.com/downloads/, then you can follow the install guide
https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio. Users should make sure that choose the c++
development tools when install Visual Studio.

Install CMake
^^^^^^^^^^^^^^^^^^^^^^^^

*If you have already installed CMake before, you can skip this step.*

Download latest(at least v3.23) package for windows from https://cmake.org/download/ and then install it. whether you choose ``.msi`` or ``.zip`` package,
you should make sure that cmake is added to your system path.

Install xLearn from Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building xLearn from source code consists two steps:

First, you need to build the executable files (``xlearn_train.exe`` and ``xlearn_predict.exe``) from the C++ code.
After that, users need to install the xLearn Python Package, which builds the extension module it needs
from the same C++ code.

Build from Source Code
=======================
Users need to clone the code from github: ::

  git clone https://github.com/aksnzhy/xlearn.git

  cd xlearn
  cmake --preset dev
  cmake --build --preset dev

Each build configuration on offer is a preset, and ``cmake --list-presets`` shows them all. The ``dev`` preset
above tunes the binaries for the machine that builds them; build the ``dist`` preset instead when the binaries
have to run on an older machine.

If the building is successful, users can find two executable files (``xlearn_train.exe`` and ``xlearn_predict.exe``) in the ``build\dev`` path,
along with the demo data. Users can test the installation by using the following command: ::

  cd build\dev
  run_example.bat

The tests are built along with the ``dev`` preset, and are run with: ::

  ctest --preset dev

Install Python Package
=======================

Then, you can install the Python package from the root of the repository: ::

  python -m pip install .

You can also test the Python package by using the following command, from the directory holding the dataset
it trains on: ::

  cd demo\classification\criteo_ctr
  python ..\..\..\python-package\test\test_python.py

Install xLearn from pip
^^^^^^^^^^^^^^^^^^^^^^^^

We provide Python package on Windows, it supports these Python(x64) versions: ``2.7, 3.4, 3.5, 3.6, 3.7``.

Users can download this binary python package from tab release_, then use ``pip`` command install the ``.whl`` file which you download.

.. _release: https://github.com/aksnzhy/xlearn/releases

After that, you can type the following script in your python shell to check whether the xLearn has been installed successfully: ::

  >>> import xlearn as xl
  >>> xl.hello()

You will see the following message if the installation is successful: ::

  -------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.43 Version --
  -------------------------------------------------------------------------

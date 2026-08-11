Installation Guide
----------------------------------

For now, xLearn can support both Linux and Mac OS X. We will support it on Windows platform in the near
future. This page gives instructions on how to build and install the xLearn using ``pip`` and how to build
it from source code. No matter what way you choose, make sure that your OS has already installed ``GCC`` or ``Clang``
(with the support of ``C++ 23``) and ``CMake`` (at least v3.23).

Install GCC or Clang
^^^^^^^^^^^^^^^^^^^^^^^^

*If you have already installed your C++ compiler before, you can skip this step.*

* On Cygwin, run ``setup.exe`` and install ``gcc`` and ``binutils``.
* On Debian/Ubuntu Linux, type the command: ::

      sudo apt-get install gcc binutils 

  to install GCC (or Clang) by using: :: 

      sudo apt-get install clang 

* On FreeBSD, type the following command to install Clang: :: 

      sudo pkg_add -r clang 

* On Mac OS X, install ``XCode`` gets you Clang.


Install CMake
^^^^^^^^^^^^^^^^^^^^^^^^

*If you have already installed CMake before, you can skip this step.*

* On Cygwin, run ``setup.exe`` and install cmake.
* On Debian/Ubuntu Linux, type the command to install cmake: ::

      sudo apt-get install cmake

* On FreeBSD, type the command: ::
   
      sudo pkg_add -r cmake

On Mac OS X, if you have ``homebrew``, you can use the command: :: 

     brew install cmake

or if you have ``MacPorts``, run: :: 

     sudo port install cmake

You won't want to have both Homebrew and MacPorts installed.

Install xLearn from Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building xLearn from source code consists two steps:

First, you need to build the executable files (``xlearn_train`` and ``xlearn_predict``) from the C++ code.
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

If the building is successful, users can find two executable files (``xlearn_train`` and ``xlearn_predict``) in
the ``build/dev`` path, along with the demo data. Users can test the installation by using the following command: ::

  cd build/dev
  ./run_example.sh

The tests are built along with the ``dev`` preset, and are run with: ::

  ctest --preset dev

Install Python Package
=======================

Then, you can install the Python package from the root of the repository: ::

  python3 -m pip install .

You can also test the Python package by using the following command, from the directory holding the dataset
it trains on: ::

  cd demo/classification/criteo_ctr
  python3 ../../../python-package/test/test_python.py

Install xLearn from pip
^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install xLearn Python package is to use ``pip``. The following command will 
download the xLearn source code from pip and install Python package locally. You must make sure that you have already installed a C++23 compiler and CMake in your local machine: ::

    sudo pip install xlearn

The installation process will take a while to complete. After that, you can type the following script in your python shell to check whether the xLearn has been installed successfully: ::

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


Install R Package
^^^^^^^^^^^^^^^^^^^^^^^^

The R package installation guide is coming soon.

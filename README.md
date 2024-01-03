# OpenVINO Verification Tester
This is a small small suite of tests to confirm proper instillation and functioning of the !!LINK!!OpenVINO AI Toolkit

The only requirements are having OpenVINO and Python 3.8 or greater and ensuring the python 3 dev headers are installed.

## Quick Start (Ubuntu/Debian) 
```
$ git clone git@github.com:bdtinc/openvino-verify.git
$ cd openvino-verify
$ sudo pip install python<major>.<minor>-dev
$ source install_linux.sh
$ ./launcher.py
```
## Manual Install
Pull down the repo. 

```
$ git clone git@github.com:bdtinc/openvino-verify.git
$ cd open-vino-verify
```

Find your python version:

`$ python3 -V`

Then make sure the proper python dev package is installed:

`$ sudo apt install python3.<MINOR_VERSION>-dev`

Create venv:

```
$ python3 -m venv "venv"
$ source venv/bin/activate
```

Unfotunately, one of the python packages we are installing doesn't have a proper manifest file, so to prevent the possiblity of pip failing due to load order, we have to install our requirements in two steps:

```
$ pip install --upgrade pip setuptools wheel
$ pip install -r build-requirements.txt
$ pip install -r requirements.txt
```

## Usage

launcher.py runs tests located in the ov-tests directory. By default, it runs a subset of five tests that will require 10-20 minutes depending on the hardware configuration of the host machine. 

To run the default tests:

`$ ./launcher.py`

### Options

There are a number of options for launcher.py.

| switch | description |
|--------|-------------|
| --info | Shows platform information and exits. |
| --device DEVICE | Limit testing to one device or type of device on the machine. Most commonly CPU or GPU.|
| --full | Runs all tests in the tests directory, by default ov-tests. |
| --run_files RUN_FILES | one or more test files (space seperated) to run. |
| --tests_dir | Alternate direcctory containing test files. |
| --tests_json | JSON file containing tests to run in the format {filename: test description}. See examples in test-sets |
| --tag TAG | Test files can be marked with a tag. Using this flag only files with the match tag will run. See the template in ov-tests for more details.

### Adding Tests

You can add tests by following the instructions in ov-tests/README.txt.




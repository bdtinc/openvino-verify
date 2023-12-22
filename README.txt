OpenVINO Verification Tester

This is a working prototype of a simple verificaton test suite for Intel's OpenVINO framework. It consists of a testrunner (launcher.py) and a set of test scripts in the ov-tests directory. The test scripts are derived from various notebooks from the openvino_notebooks repository (https://github.com/openvinotoolkit/openvino_notebooks).

INSTALLING

Requirements

We assume you already have OpenVINO 2022.3 or above and Python 3.8+ installed and working on your machine.

You should set up a python virtual environment to run this code:
[need instructions for this here]

Place all the files from the repo in your virtual environment. 

In order to install the specific python libraries used, run the following commands:
$ pip install -r build-requirements.txt
$ pip install -r requirements.txt

RUNNING THE TEST SUITE 

The basic use is very simple:

$ python launcher.py 

By default, launcher will run a default set of the test files in the ov-test subdirectory once for each available OpenVINO-visible device on your machine. As it runs it will produce a table showing you the results of each test. When it is finished a test_results json file will be produced to save the results of your tests.

There are a few options to select tests to be run.
--full will run the full set of tests located in the test directory
--device DEVICE  will only run the tests against a single OpenVINO-visible device
--run_files FILE1 FILE2 runs one or more specific test files, these are relative to /ov-tests 
--tests_json JSON_FILE this runs a set of tests determined by a json file. Files for single, small, full, fail and pass are included in the test-sets directory.

The default set of tests will take 10-20 minutes to run depending on your installed devices, internet speed and processor.

The full set of tests can take up to 40 minutes, again depending.

Repeated runs are much quicker since most assets are preserved between runs. 

TODO:

- Fix this doc, convert to md. 
- Add a developer doc to discuss adding tests?
- need an --info switch that just shows the platform information headers but doesn't run the tests.
- look at adding typehints to launcher.py, see if I need to clean up those nested logic loops for clarity.
- add either extra script or a launcher.py switch to clean up the downloads
- script to set up the virtual env? Maybe instructions are good enough, dunno.
- !! add python3-dev requirement to docs !!
- make sure setuptools and wheel pip install are in virtenv setup section




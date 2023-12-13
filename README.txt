OpenVINO Verification Tester

This is a working prototype of a simple verificaton test suite for Intel's OpenVINO framework. It consists of a testrunner (launcher.py) and a set of test scripts in the ov-tests directory. The test scripts are derived from various notebooks from the openvino_notebooks repository (https://github.com/openvinotoolkit/openvino_notebooks).

INSTALLING

Requirements

We assume you already have OpenVINO 2022.3 or above and Python 3.8+ installed and working on your machine.

You should set up a python virtual environment to run this code:
[need instructions for this here]

Place all the files from the repo in your virtual environment. Right now these are being distributed as a downloaded bundle from a private folder, but these should end up in a git repo later.

In order to install the specific python libraries used, run the following command:
$ pip install -r requirements.txt

RUNNING THE TEST SUITE 

The basic use is very simple:

$ python launcher.py 

By default, launcher will go through every python file in the ov-test subdirectory and run it once for each available OpenVINO-visible device on your machine. As it runs it will produce a table showing you the results of each test. When it is finished a test_results json file will be produced to save the results of your tests.

At the moment there are only five active tests and on my 11th Gen Core i7 with an integrated Iris GPU it takes around 15 minutes to run through from a standing start. Repeated runs are quicker since most assets are preserved between runs. 

There are a few options to select tests to be run.
--device DEVICE  will only run the tests against a single OpenVINO-visible device
--run_files FILE1 FILE2 runs one or more specific test files, these are relative to /ov-tests 
--tests_json JSON_FILE this runs a set of tests determined by a json file. Files for single, small, full, fail and pass are included in the base directory.

Before we finish will probably have a pair of switches --fast and --complete to run a small but reasonably rapid subset of tests or run every test including some that we might have decided are too long to make default. I want to add machinery to have tags in the source files to control group inclusion for these or other switches, but that doesn't exist yet. Maybe better sticking with the json files, not sure.

Also need an --info switch that just shows the platform information headers, but doesn't run the tests.





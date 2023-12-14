#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# environment needs yaspin

import argparse
import os
from pathlib import Path
import subprocess
import sys
import json
import datetime
import platform
import openvino as ov
from yaspin import yaspin

TEST_DIR = './ov-tests'

def show_platform_information():
    sysinfo = {}
    sysinfo['platform'] = platform.platform()
    sysinfo['kernel_version'] = os.uname().release if os.name == 'posix' else platform.uname().release
    sysinfo['python_version'] = platform.python_version()
    sysinfo['openvino_version'] = ov.__version__
    sysinfo['available_devices'] = {}
    core = ov.Core()
    for device in core.available_devices:
        sysinfo['available_devices'][device] = core.get_property(device, 'FULL_DEVICE_NAME')

    print("\n")
    print(f"{'Platform':20}: {sysinfo['platform']}")
    print(f"{'Kernel Version':20}: {sysinfo['kernel_version']}")
    print(f"{'Python Version':20}: {sysinfo['python_version']}")
    print(f"{'OpenVINO Version':20}: {sysinfo['openvino_version']}")
    print("\n")

    core = ov.Core()
    gpu_flag = False
    print(f"Available Hardware:")
    for device in sysinfo['available_devices']:
        print(f"{device:10} {sysinfo['available_devices'][device]}")
        if 'GPU' in device: gpu_flag = True
    if not gpu_flag:
        print(f"{'GPU':10} {'None Found'}")
    if not 'NPU' in core.available_devices:
        print(f"{'NPU':10} {'None Found'}")
    return(sysinfo)

def print_result_line(tests, test_file, test_results):
    RED = '\033[91m'
    GREEN = '\033[92m'
    CLOSE = '\033[0m'
    core = ov.Core()
    results_line = f"{tests[test_file]:<50}"
    for device in core.available_devices:
        if device not in test_results[test_file]:
            test_results[test_file][device] = 'N/A'
        if test_results[test_file][device] == 'PASS':
            results_line += f" {GREEN}{test_results[test_file][device]:6}{CLOSE}"
        elif test_results[test_file][device] == 'FAIL':
            results_line += f" {RED}{test_results[test_file][device]:6}{CLOSE}"
        else:
            results_line += f" {test_results[test_file][device]:6}"
    print(results_line)
    return

def test_header_line():
    core = ov.Core()
    header = f"{'Test':<50}"
    underline = f"{'-'*50:<50}"
    for device in core.available_devices:    
        header += f" {device:6}"
        underline += f" {'-'*6:6}"
    print(header)
    print(underline)
    return

def get_tests_from_dir(test_dir):
    tests = {}
    for test_file in os.listdir(test_dir):
        if test_file.endswith('.py'):
            with open(os.path.join(test_dir, test_file), 'r') as f:
                for line in f:
                    if line.startswith('# Test_Slug_Line:'):
                        test_name = line.split(':')[1].strip()
                        tests[test_file] = test_name
                        break
    return tests

def get_tests_from_list(test_list, test_dir):
    tests = {}
    for test_file in test_list:
        if test_file.endswith('.py'):
            with open(os.path.join(test_dir, test_file), 'r') as f:
                for line in f:
                    if line.startswith('# Test_Slug_Line:'):
                        test_name = line.split(':')[1].strip()
                        tests[test_file] = test_name
                        break
    return tests

def run_test(test_file, device='AUTO'):
    cmd = f"python {test_file} --device {device}"
    result = ''
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        returncode = 0
    except subprocess.CalledProcessError as e:
        result = e.output
        returncode = e.returncode
    if returncode == 0: status = 'PASS'
    # 13 is the return code we are using for a test/device combo that is not supported
    elif returncode == 13: status = 'N/A'
    else: status = 'FAIL'
    return(status, result)

def get_test_devices(device=None):
    core = ov.Core()
    test_devices = []
    # get all gpu devices and not just GPU.0 or GPU.1
    if device == 'GPU':
        for avialable_device in core.available_devices:
            if 'GPU' in avialable_device:
                test_devices.append(avialable_device)
        if not test_devices:
            print(f"ERROR: {device} is not an available device.")
            sys.exit(1)
    elif device == 'CPU':
        test_devices.append('CPU')
    elif device == 'NPU':
        if 'NPU' in core.available_devices:
                test_devices.append('NPU')
        else:
            print(f"ERROR: {device} is not an available device.")
            sys.exit(1)
    # catch specific gpu or npu callout if multiple devices are available
    elif device is not None:
        if device in core.available_devices:
            test_devices.append(device)
        else:
            print(f"ERROR: {device} is not an available device.")
            sys.exit(1)
    # default to all available devices
    else:
        test_devices = core.available_devices

    return test_devices

def run_tests(tests, test_dir, device=None):
    test_devices = get_test_devices(device)
    test_results = {}
    test_failures = {}
    for test_file in tests:
        test_results[test_file] = {}
        test_path = os.path.join(test_dir, test_file)
        with yaspin(text=f"Running {tests[test_file]}", color="yellow") as sp:
            for device in test_devices:
                status, result = run_test(test_path, device)
                test_results[test_file][device] = status
                if status == 'FAIL':
                    if test_file not in test_failures: test_failures[test_file] = {}
                    test_failures[test_file][device] = result
                    
        print_result_line(tests, test_file, test_results)
    return (test_results, test_failures)

def dump_errors(failures, filename):
    with open(filename, 'w') as f:
        for test in failures:
            f.write(f"{test}\n")
            for device in failures[test]:
                f.write(f"{device}: \n")
                f.write(f"{failures[test][device]}\n")
    return

def dump_results(sys_info, results, failures, filename):
    export_dict = {}
    export_dict['Platform Info'] = sys_info
    export_dict['Tests'] = results
    if failures:
        export_dict['Errors'] = failures
    json_data = json.dumps(export_dict, indent=4)
    with open(filename, 'w') as f:
        f.write(json_data)
    print(f"Test results written to {filename}")
    return

def main(tests, test_dir, device=None):    
    #build results_json_filename
    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    results_json_filename = f"test_results_{date_time}.json"

    sys_info = show_platform_information()

    print('\n')
    test_header_line()
    results, failures = run_tests(tests, test_dir, device)
    print('\n')
    if failures:
        print('\n')
        print('Failures in test file(s): ')
        for test in failures: print(f"{test}")
        print('\n')
        print(f"Full error messages in ERRORS.txt and Errors section of {results_json_filename}") 
        dump_errors(failures, 'ERRORS.txt')
    
    dump_results(sys_info, results, failures, results_json_filename)

if __name__ == '__main__':
    tests = {}
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests_dir', help='Directory containing test files', default=TEST_DIR)
    parser.add_argument('--device', help='Device to run tests on', default=None)
    parser.add_argument('--full', help='Run all tests', action='store_true')
    parser.add_argument("--run_files", nargs='+', help="List of test filenames to run. \
                                                        Files are relative to TESTS_DIR.")
    parser.add_argument("--tests_json", help="JSON file containing tests to run  \
                                            in the format {filename: test description}")
    args = parser.parse_args()

    if args.run_files:
        tests = get_tests_from_list(args.run_files, args.tests_dir)
    elif args.tests_json:
        with open(args.tests_json, 'r') as f:
            tests = json.load(f)
    if not tests:
        tests = get_tests_from_dir(args.tests_dir)
    
    main(tests, args.tests_dir, device=args.device)

Creating New Tests 
-------------------------

To add tests to the test suite, copy ov-tests/test-template.txt to ov-tests/<filename>.py.

A few notes about the template:

- You can create tagged groups of tests by adding tags to the # Test_Groups: line. Tags should be space seperated. Do not uncomment the line.

- The imports and boilerplate included before "if __name__ == '__main__':" are necessary to to set the proper paths to the data and models directories as well as importing some download utilities used by almost all tests. Leave this intact. 

- The signature of download_file is (url, filename, directory) where url is the url with the resource to be downloaded, filename is the save name and directory is the directory where you want the file saved.
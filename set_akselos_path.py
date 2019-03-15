# Copyright (C) 2015 Akselos
import sys
import json
import os

'''
This is a simple script that finds the akselos repository directory by walking up the directory
tree until it finds akselos_config.json (similar to how hg finds the .hg directory), or from the
AKSELOS_DIR environment variable.  This script then sets the Python path to include the tools
directory corresponding to the akselos repository. Note that an important property of this script is
that different forks or instances of the akselos installation can share it
(i.e. this script only needs to be installed only once per computer). 
This script should not be changed in a way that will break backwards compatibility, so older
installations can still run.
'''

def get_akselos_dir():
    akselos_dir = os.environ.get('AKSELOS_DIR', None)
    if akselos_dir is None:
        # Note that we use the PWD environment variable, if available, to get the current directory.
        # This variable is typically set by the user's shell, and contains the logical directory.
        # This can be different than os.abspath('.') in the case that we followed a symlink to
        # get to the current directory.  In practice this may happen when the data directory is
        # shared between more than one akselos repository; in this case using os.path.abspath('.')
        # will confusingly give us the akselos repository based on the physical directory layout,
        # which is not necessarily the one would expect based on the shell 'pwd' command.
        # We also check '.' if PWD does not work because pycharm debugger sets PWD to the home
        # directory for some annoying reason.
        pwd = os.environ.get('PWD', None)
        if pwd is not None:
            start_paths = [pwd, os.path.abspath('.')]
        else:
            start_paths = [os.path.abspath('.')]

        for start_path in start_paths:
            akselos_dir = start_path
            failed = False
            while not os.path.isfile(os.path.join(akselos_dir, 'akselos_config.json')):
                old_path, akselos_dir = akselos_dir, os.path.dirname(akselos_dir)
                if akselos_dir == old_path:
                    failed = True
                    break
            if not failed:
                break

        if failed:
            raise ValueError('Could not determine akselos repository because '
                             'akselos_config.json is not in a parent directory of the '
                             'current directory, and the AKSELOS_DIR environment variable '
                             'is not set.')
    return akselos_dir

akselos_dir = get_akselos_dir()
scripts_dir = os.path.join(akselos_dir, 'tools') 
if not scripts_dir in sys.path:
    sys.path.insert(0, scripts_dir)

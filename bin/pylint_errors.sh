#!/bin/bash
# Make it so that any command that fails in here also fails the script and causes a build error
set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $script_dir/..

find bin tests src -name "*.py" | xargs pylint

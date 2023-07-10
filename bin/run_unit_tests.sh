#!/bin/bash
# Make it so that any command that fails in here also fails the script and causes a build error
set -e

# Parse the command line arguments
while getopts ":b:" opt; do
  case $opt in
    b) branch_name="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

echo "Running unit tests on branch $branch_name"

if test -f venv/bin/activate; then
    . venv/bin/activate
fi

# Set the environment variable for the service account key
pytest -m "not slow"

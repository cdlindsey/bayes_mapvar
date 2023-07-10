''' Configuration for Unit Testing '''
import pytest
# Extra configuration for unit testing.
# for testing graph mode,
#   python -m pytest -s --rungraph -m "not eager"
# for testing eager,
#   python -m pytest -s
# you can test graph and some of eager by running
#   python -m pytest -s --rungraph
#   Eager tests will exit out immediately after Graph tests have been performed.


def pytest_addoption(parser):
    ''' Adds options for testing. '''
    parser.addoption("--rungraph",
                     action="store_true",
                     default=False,
                     help="run TensorFlow Graph Execution Tests")


def pytest_configure(config):
    ''' Testing configuration. '''
    config.addinivalue_line(
        "markers", "graph: mark test as using Graph Execution for TensorFlow")


def pytest_collection_modifyitems(config, items):
    ''' Setup so that markers can be skipped. '''
    if config.getoption("--rungraph"):
        # --rungraph given in cli: do not skip tests that use Graph execution for Tensorflow
        return
    skip_graph = pytest.mark.skip(reason="need --rungraph option to run")
    for item in items:
        if "graph" in item.keywords:
            item.add_marker(skip_graph)

import imp
import importlib
import os


def load(name):
    pathname = os.path.join(os.path.dirname(__file__), name)
    return imp.load_source('', pathname)

# def load(name):
#     pathname = os.path.join(os.path.dirname(__file__), name)
#     spec = importlib.util.spec_from_file_location('', pathname)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module
# -*- coding: utf-8 -*-

# Copyright (c) 2022-2023 by Jesse-James PRINCE AGBODJAN for SRD-Energies (https://www.srd-energies.fr/) 
# All rights reserved.

# This File MUST be located in the '../Modules/' folder.

# fnfrnb : function from notebook

# This module is heavely inspired of https://jupyter-notebook.readthedocs.io/en/v6.5.2/examples/Notebook/Importing%20Notebooks.html

# Im using this since, python's direct scipt (.py) does not allow using ipython magic commands 
# (% ans %%) such as the one use to run par opf in :func:par_block_pfOpf( ): 


""" Load a function from a Jupyter notebooks """

import io, os, sys, types
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell


def find_notebook(fullname, path=None):
    """find a notebook, given its fully qualified name and an optional path

    This turns "foo.bar" into "foo/bar.ipynb"
    and tries turning "Foo_Bar" into "Foo Bar" if Foo_Bar
    does not exist.
    """
    name = fullname.rsplit('.', 1)[-1]
    if not path:
        path = ['']
    for d in path:
        nb_path = os.path.join(d, name + ".ipynb")
        if os.path.isfile(nb_path):
            return nb_path
        # let import Notebook_Name find "Notebook Name.ipynb"
        nb_path = nb_path.replace("_", " ")
        if os.path.isfile(nb_path):
            return nb_path
        
        

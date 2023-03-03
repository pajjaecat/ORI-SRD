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

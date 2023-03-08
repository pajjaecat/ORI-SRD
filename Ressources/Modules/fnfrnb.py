# -*- coding: utf-8 -*-

# Copyright (c) 2022-2023 by Jesse-James PRINCE AGBODJAN for SRD-Energies (https://www.srd-energies.fr/) 
# All rights reserved.

# This File MUST be located in the '../Modules/' folder.

# fnfrnb : function from notebook

# This module is heavely inspired of https://jupyter-notebook.readthedocs.io/en/v6.5.2/examples/Notebook/Importing%20Notebooks.html

# I'm using this since, python's direct script (.py) does not allow using ipython magic commands
# (% and %%) such as the one use to run par opf in :func:par_block_pfOpf:


""" Load a function from a Jupyter notebooks """

import io
import os
import sys
import types
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell


def find_notebook(fullname: str, path=None):
    """find a notebook, given its fully qualified name and an optional path.

    This turns "foo.bar" into "foo/bar.ipynb"
    and tries turning "Foo_Bar" into "Foo Bar" if Foo_Bar
    does not exist.

    Parameters
    ----------
    fullname : str
        Name of the jupyter notebook, without the extension `ipynb`.
    path : str, Optional, Default=None
        Relative location of notebook to load.

    Returns
    -------
    Notebook path.

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


class NotebookLoader(object):
    """Module Loader for Jupyter Notebooks

    Parameters
    ----------
    path : str, Optional, Default=None
        Relative location of notebook to load.

    Attributes
    ----------
    shell :
        InteractiveShell.instance()
    path : str
        Relative location of notebook to load.

    """

    def __init__(self, path=None):
        """Module Loader for Jupyter Notebooks

        Parameters
        ----------
        path : str, Optional, Default=None
            Relative location of notebook to load.

        """

        self.shell = InteractiveShell.instance()
        self.path = path

    def load_module(self, fullname):
        """import a notebook as a module

        Parameters
        ----------
        fullname: str
            Name of the jupyter notebook, without the extenxion `ipynb`.

        """
        path = find_notebook(fullname, self.path)

        print(f"Importing {fullname}.ipynb content as a module")

        # load the notebook object
        with io.open(path, 'r', encoding='utf-8') as f:
            nb = read(f, 4)

        # create the module and add it to sys.modules
        # if name in sys.modules:
        #    return sys.modules[name]
        mod = types.ModuleType(fullname)
        mod.__file__ = path
        mod.__loader__ = self
        mod.__dict__['get_ipython'] = get_ipython
        sys.modules[fullname] = mod

        # extra work to ensure that magics that would affect the user_ns
        # actually affect the notebook module's ns
        save_user_ns = self.shell.user_ns
        self.shell.user_ns = mod.__dict__

        try:
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    # transform the input to executable Python
                    code = self.shell.input_transformer_manager.transform_cell(cell.source)
                    # run the code in the module
                    exec(code, mod.__dict__)
        finally:
            self.shell.user_ns = save_user_ns
        return mod

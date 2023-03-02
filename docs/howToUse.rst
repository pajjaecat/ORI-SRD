  
How to Use
===============

This section gives an in-depth view of how to efficiently navigate and use the `Ori's Github Repository <https://github.com/pajjaecat/ORI-SRD>`_.


Set up
------------


Download the repository
^^^^^^^^^^^^^^^^^^^^^^^^

The repository is available  `here <https://github.com/pajjaecat/ORI-SRD>`_. After downloading, make sure to check out its `Architecture <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/README.md>`_ to comprehend how the files are organized. 



Create  new environment 
^^^^^^^^^^^^^^^^^^^^^^^^

All the code presented here is written and tested with `python3.7 <https://www.python.org/>`_, making use of the powerful `Jupyter <https://jupyter.org/)>`_. To reproduce, please follow the instructions `here <https://stackoverflow.com/questions/48787250/set-up-virtualenv-using-a-requirements-txt-generated-by-cond>`_ to create a new environment. Use either of files `environment.yml <https://github.com/pajjaecat/ORI-SRD/blob/main/environment.yml>`_ or  `requirement.txt <https://github.com/pajjaecat/ORI-SRD/blob/main/requirements.txt>`_  or both to do so. These files contain all the necessary packages and libraries (from the relevant channels) to use to avoid compatibility issues.



Some Definitions
^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 25 50
   :header-rows: 1

   * - Definitions
     - Meaning
   * - **LV\lv Prod\Sgens**
     - Lower voltage generators (producteurs BT)
   * - **HV\hv Prod or Sgens**
     - High voltage generators (producteurs HTA)
   * - **Upper Network**
     - The network where is located the Main Poste Source
   * - **Lower Network**
     - The network to study which is a small branch of the Upper Network

By default, we consider *ST LAURENT* and *CIVAUX* as, respectively, the upper and lower Network.


.. warning:: 
     Please **READ** |vRiseBlockScheme|_  before reading everything else.

.. |vRiseBlockScheme| replace:: `The voltage rise detection block scheme`
.. _vRiseBlockScheme: https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Docs/VRiseControlBlockScheme.pdf
.. |uppernet| replace:: `ST LAURENT`
.. |lowernet| replace:: `CIVAUX`



How to Use
##############
 
 
This section gives an in-depth view of how to efficiently navigate and use the `Ori's Github Repository <https://github.com/pajjaecat/ORI-SRD>`_.


Set Up Your System
--------------------
**********************


Download the repository
^^^^^^^^^^^^^^^^^^^^^^^^

The repository is available  `here <https://github.com/pajjaecat/ORI-SRD>`_. After downloading, make sure to check out its `Architecture <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/README.md>`_ to comprehend how the files are organized. 



Create  new environment 
^^^^^^^^^^^^^^^^^^^^^^^^

All the code presented here is written and tested with `python3.7 <https://www.python.org/>`_, making use of the powerful `Jupyter <https://jupyter.org/)>`_. To reproduce, please follow the instructions `here <https://stackoverflow.com/questions/48787250/set-up-virtualenv-using-a-requirements-txt-generated-by-cond>`_ to create a new environment. Use either of files `environment.yml <https://github.com/pajjaecat/ORI-SRD/blob/main/environment.yml>`_ or  `requirement.txt <https://github.com/pajjaecat/ORI-SRD/blob/main/requirements.txt>`_  or both to do so. These files contain all the necessary packages and libraries (from the relevant channels) to use to avoid compatibility issues.


Some Definitions
^^^^^^^^^^^^^^^^^^

.. list-table:: Variables Used in code. 
   :widths: 25 50
   :header-rows: 1

   * - Definitions
     - Meaning
   * - **LV\\lv Prod\\Sgens**
     - Lower voltage generators (producteurs BT)
   * - **HV\hv Prod or Sgens**
     - High voltage generators (producteurs HTA)
   * - **Upper Network**
     - The network where is located the Main Poste Source
   * - **Lower Network**
     - The network to study which is a small branch of the Upper Network

By default, we consider |uppernet| and |lowernet| as, respectively, the upper and lower Network. 


.. warning:: 
     Please **READ** |vRiseBlockScheme|_ .
     
  
Add Files
^^^^^^^^^^^^

.. note::
   This step is only necessary if one wants to apply Ori to networks different  from |uppernet| and |lowernet|.
 
 
- Add networks (lower and upper) files to  `Pickle_files <https://github.com/pajjaecat/ORI-SRD/tree/main/Ressources/Pickle_files>`_;
- Add the networks' input files to `Excel_files <https://github.com/pajjaecat/ORI-SRD/tree/main/Ressources/Excel_files>`_. 

   - Each of the Hv Prod must have its own associated file;
   - All the Lv Prods (i.e. all the LV Prods in the Upper network) must be aggregated in a unique file;
   - All the Load (i.e. all the load in the Upper network) must be aggregated in a unique file.




Apply the |vRiseBlockScheme|_ 
-------------------------------
**********************************


To apply the |vRiseBlockScheme|_, we propose the following steps. 


Define variables
^^^^^^^^^^^^^^^^^^

The goal here is to define all the default variables to use for all simulations in a file. All the necessary variables are described in :ref:`defaultVariables` and can be modified there. Remember that modifying a variable's value will affect all the modules and notebooks using the variable. 


Clean your data
^^^^^^^^^^^^^^^^^
Cleaning the network's input data is an essential part of the process. In this case, there is no step-by-step to follow because the data might vastly vary from one problem to another. An example of how one might apply this process is available in `CleanDataSTLaurentDeJourdes`_ :ref:`RstCleanDataSTLaurentDeJourdes`. 


Run Simulations
^^^^^^^^^^^^^^^^
The simulations can be run for different models of the prediction block **PRED**. This section covers three of them, namely 

#. "Future Known" , `Future Known`_
#. Persistence;
#. Recurrent Neural Network.

The first two are easily implementable, while the third is more complex (Advanced User).

Future Known
==============
"RstCleanDataSTLaurentDeJourdes", ``RstCleanDataSTLaurentDeJourdes``


Persistence
============



Recurrent Neural Network
========================







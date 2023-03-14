.. |vRiseBlockScheme| replace:: `voltage rise detection block scheme`
.. _vRiseBlockScheme: https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Docs/VRiseControlBlockScheme.pdf
.. |uppernet| replace:: `ST LAURENT`
.. |lowernet| replace:: `CIVAUX`



How to Use
##############
 
 
This section gives an in-depth view of how to efficiently navigate and use the
`Ori's Github Repository <https://github.com/pajjaecat/ORI-SRD>`_.

.. warning:: 
    Please **READ** the |vRiseBlockScheme|_ .
  


Set Up Your System
--------------------
**********************


Download the repository
^^^^^^^^^^^^^^^^^^^^^^^^

The repository is available  `here <https://github.com/pajjaecat/ORI-SRD>`_. After downloading, make sure to check out
its `Architecture <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/README.md>`_ to comprehend how the files
are organized.



Create  new environment 
^^^^^^^^^^^^^^^^^^^^^^^^

All the code presented here is written and tested with `python3.7 <https://www.python.org/>`_, making use of the
powerful `Jupyter <https://jupyter.org/)>`_. To reproduce, please follow the instructions
`here <https://stackoverflow.com/questions/48787250/set-up-virtualenv-using-a-requirements-txt-generated-by-cond>`_ to
create a new environment. Use either of files `environment.yml <https://github.com/pajjaecat/ORI-SRD/blob/main/environment.yml>`_
or  `requirement.txt <https://github.com/pajjaecat/ORI-SRD/blob/main/requirements.txt>`_  or both to do so. These files
contain all the necessary packages and libraries (from the relevant channels) to use to avoid compatibility issues.


Some Definitions
^^^^^^^^^^^^^^^^^^
We define in this section some Variables used in the code and bellow.

============================  =====================================================================================
         **Definitions**                **Meaning**
============================  =====================================================================================
**LV\\lv Prod\\Sgens**         Lower voltage generators (producteurs BT)
**HV\hv Prod or Sgens**        High voltage generators (producteurs HTA)
**Upper Network**              The network where is located the Main Poste Source
**Lower Network**              The network to study, which is a small branch of the Upper Network
**Voltage Rise**               Elevation de tension
**Voltage Rise event**         | An event where the value of the  maximum voltage rise :math:`vm\_ pu\_ max` detected
                               | at the output of block **PF** is above that of the maximum authorised threshold  
                               | :math:`defAuth\_ hvBus\_ V_{rise}^{max}` (defined as :data:`oriVariables.defAuth_hvBus_vRiseMax`) 
                                  given
                               | as input to block **PF/OPF** .
============================  =====================================================================================


By default, we consider |uppernet| and |lowernet| as the upper and lower Network, respectively. 


Add Files
^^^^^^^^^^^^

.. note::
   This step is only necessary if one wants to apply Ori to networks different from |uppernet| and |lowernet|.
   Otherwise, all the necessary files to smoothly run simulations for the previously mentioned networks are
   already present in the repository.
 
 
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

The goal here is to define all the default variables to use for all simulations in a file. All the necessary variables
are described in :ref:`defaultVariables` and can be modified there. Remember that modifying a variable's value will
affect all the modules and notebooks using the variable.


Clean your data
^^^^^^^^^^^^^^^^^
Cleaning the network's input data is an essential part of the process. In this case, there is no step-by-step to follow
because the data might vastly vary from one problem to another. An example of how one might apply this process is
available in :ref:`RstCleanDataSTLaurentDeJourdes`.


Run Simulations
^^^^^^^^^^^^^^^^
The simulations can be run with different models of the prediction block **PRED**. This section covers three of them,
that are

#. `Future Known`_
#. `Persistence`_;
#. `Recurrent Neural Network`_.

The first two are easily implementable, while the third is more complex (Advanced Users). For each case, we provide a
corresponding tutorial (using the default networks ST LAURENT and CIVAUX ) from which inspiration can be drawn. Check
out section :ref:`Tutorials`, for a succinct list of all available tutorials.

Future Known
=============
For comparison purposes and to establish a baseline, the simulations must first be run, supposing the prediction block 
**PRED** has perfect knowledge of the future. See :ref:`Rst2021_2022_KnownFuture` for the associated tutorials.


Persistence
===========
The second prediction model proposed is the previous period persistence model. Tutorials 
:ref:`Rst2021_2022_Persistence` present its usage for two different values of :math:`defAuth\_ hvBus\_ V^{max}_{rise}`.

Compared to the first case (i.e., :math:`defAuth\_ hvBus\_ V^{max}_{rise} = 1.0250`), the second 
(i.e., :math:`defAuth\_ hvBus\_ V^{max}_{rise} = 1.0225`) is provided to show how the total number of voltage rise events 
could be reduced at a price of less yearly energy injection by the controllable Hv Prod. 

To implement the robust method introduced in section 2 of |vRiseBlockScheme|_, we also provide :ref:`Rst2021_2022_PersistenceRob`.


Recurrent Neural Network
========================
The last prediction model implemented is a Recurrent Neural Network (RNN). 

  .. warning ::
       All the RNNs developed in the :ref:`Tutorials` are tailored for the default networks. Using the same RNN architecture on others
       networks might not yield the best performance. We **strongly** recommend optimizing your RNN architecture depending on 
       your networks and the input data. 
       
We proposed creation, training and usage of three diferents RNN architecture that can be used solely or combined. 


Go further
-----------
**************
This section presents extensive tutorials based on the one presented in the previous section and their saved results. Please read the previous 
section `Apply the |vRiseBlockScheme|_ ` before diving here.




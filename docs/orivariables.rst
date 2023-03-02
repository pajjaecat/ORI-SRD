.. _defaultVariables:


#############################
     Default variables
#############################
This section lists all the default variables used in the different `notebooks <https://github.com/pajjaecat/ORI-SRD/tree/main/Ressources/Notebooks>`_. You may changed them globally in `oriVariables <https://github.com/pajjaecat/ORI-SRD/blob/1.0.2/Ressources/Modules/oriVariables.py>`_ according the problem specifities. 



.. automodule:: oriVariables 
  :members:
  :undoc-members:
  :show-inheritance:
  
  
.. code:: python 

     attr_list = [('bus', 'name'),
                  ('load', 'bus'),
                  ('switch', 'bus'),
                  ('line', 'from_bus'),
                  ('line', 'to_bus'),
                  ('trafo', 'hv_bus'),
                  ('trafo', 'lv_bus')]
     """list :  Atribute list """

     network_folder = '../Pickle_files/'
     """str : Relative location of the network to study. """

     excel_folder = '../Excel_files/'
     """str : Relative location of the network to study input's. """

     py_folder = '../Modules/'
     """str: relative location of the modules to be use."""

     Δt = 1/6  # Time frequency 10mn ==> 1Hour/6
     """float:  Time frequency,  10mn ==> 1Hour/6"""



     defAuth_hvBus_vRiseMax = 1.025
     """float : Default max authorised voltage rise on Hv Buses on Lower network."""

     defAuth_hvBus_vRiseMin = 0.95
     """float : Default min authorised voltage rise on Hv Buses on Lower network."""

     defAuth_lvBus_vRiseMax = 1.075
     """float : Default max authorised voltage rise on Lv Buses on Lower network."""

     defAuth_lvBus_vRiseMin = 0.95
     """float :  Default min authorised voltage rise on Lv Buses on Lower network."""

     default_hv_voltage = 20.6
     """float : Default voltage (kV) of Hv buses on Lower Network."""

     default_lv_voltage = 0.4
     """float : Default voltage (kV) of Lv buses on Lower Network. """

     default_ctrld_hvProd_max = 4.0
     """float : Default maximum output (MW ) of te controlled Hv producer. """

     h_start_end = ('07:00','18:50')
     """tuple : Daylight period."""

     trainVal_split_date = '2021 12 31 23:50'  # Date of training+Validation split data Lower bond
     """str : Training + validation set split date """

     train_split_date = '2021 06 01'        # lower date to split training and validation data
     """str : Training set split date """

     testSet_end_date = '2022 06 02'
     """str : Test set end split data """

     testSet_start_date = '2021 06 03'
     """str : Test set start split data """

     testSet_start_date_M2 = '2021 06 01'
     """str : Test set start split data minus two days """


# -*- coding: utf-8 -*-

# Copyright (c) 2022-2023 by Jesse-James PRINCE AGBODJAN for SRD-Energies (https://www.srd-energies.fr/) 
# All rights reserved.

""" List of all defaut Variables used in the tutorials present in `~/Ressources/Notebooks/ <https://github.com/pajjaecat/ORI-SRD/tree/main/Ressources/Notebooks>`_ .

This module holds a list of all the variables used in the ORI package.



"""

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
"""str: relative location of the modules to be use.

Notes
-----
    This file MUST be present in the :py:data:`oriVariables.py_folder`

"""

Δt = 1/6  # Time frequency 10mn ==> 1Hour/6
"""int:  Time frequency"""


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


train_split_date = '2021 12 31 23:50' # Date of training+Validation split data Lower bond 
trainVal_split_date = '2021 06 01'     # lower date to split training and validation data
testSet_end_date = '2022 06 02'
testSet_start_date = '2021 06 03'



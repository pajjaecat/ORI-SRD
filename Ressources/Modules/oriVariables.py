# -*- coding: utf-8 -*-

# Copyright (c) 2022-2023 by Jesse-James PRINCE AGBODJAN for SRD-Energies (https://www.srd-energies.fr/) 
# All rights reserved.

# This File MUST be located in the '../Modules/' folder.

"""This module holds a list of all the variables used in the ORI package.


"""

# ------------------          Folders locations     -----------------------------#

network_folder = '../Pickle_files/'
"""str : Relative location of the network to study. """

excel_folder = '../Excel_files/'
"""str : Relative location of the network to study input's. """

modules_folder = '../Modules/'
"""str: relative location of the modules to be use."""

simResult_folder = f'{modules_folder}simulationResults/'
"""str: relative location of the folder containing all the simulation result."""



# -----------------------   Default Values to apply to networks  -----------------#

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


#--------------------   Spliting data into several periods ------------------------#
# These dates must be all included and defined depending on the date in the network
# input data. As preference,  it is advised to use the closest data  for the prediction
# i.e, previous year for prediction, 6 months before the closest as validation and year
# before that as training


trainVal_split_date = '2021 12 31 23:50'  # Date of training+Validation split data Lower bond
"""str : Training + validation set split date, lower Bound """

train_split_date = '2021 06 01'        # lower date to split training and validation data
"""str : Training set split date """

testSet_start_date = '2021 06 03'
"""str : Test set start split data """

testSet_start_date_M2 = '2021 06 01'
"""str : Test set start split data minus two days """

testSet_start_date_M1 = '2021 06 02'
"""str : Test set start split data minus one day """

testSet_end_date = '2022 06 02'
"""str : Test set end split data """

testSet_end_date_M1 = '2022 06 01'
"""str : Test set end split data """


# ************** Training set
trainSet_start = '2020 01 01'
"""str : Training set start date. """

trainSet_end = '2021 06 02 23:50'
"""str : Training set end date """

trainSet_end_M1 = '2021 06 01 23:50'
"""str : Training set end date minus one day """

trainSet_end_M2 = '2021 05 31 23:50'
"""str : Training set end date minus two days """

# *************** Validation Set
valSet_start = '2021 06 03'
"""str : Validation set start date """

valSet_start_M1 = '2021 06 02'
"""str : Validation set start date minus one day """

valSet_start_M2 = '2021 06 01'
"""str : Validation set start date minus two days """

valSet_end = '2022 01 02 23:50'
"""str : Validation set end date """

valSet_end_M1 = '2022 01 01 23:50'
"""str : Validation set start date minus one day """

valSet_end_M2 = '2021 12 31 23:50'
"""str : Validation set start date minus two days """

# ****************** Test Set
testSet_start = '2022 01 03'
"""str : Test set start date """

testSet_start_M1 = '2022 01 02'
"""str : Test set start date minus one day """

testSet_start_M2 = '2022 01 01'
"""str : Test set start date minus two days """

testSet_end = '2022 06 02 23:50'
"""str : Test set end date """

testSet_end_M1 = '2022 06 01 23:50'
"""str : Test set start date minus one day """

testSet_end_M2 = '2022 05 30 23:50'
"""str : Test set start date minus two days """



#--------------------   Others variables ------------------------#

Δt = 1/6  # Time frequency 10mn ==> 1Hour/6
"""float:  Time frequency,  10mn ==> 1Hour/6"""

attr_list = [('bus', 'name'),
             ('load', 'bus'),
             ('switch', 'bus'),
             ('line', 'from_bus'),
             ('line', 'to_bus'),
             ('trafo', 'hv_bus'),
             ('trafo', 'lv_bus')]
"""list :  Atribute list """


h_start_end = ('07:00','18:50')
"""tuple : Daylight period."""


h10_start_end = ('07:10','18:50')
"""tuple : Daylight period ."""



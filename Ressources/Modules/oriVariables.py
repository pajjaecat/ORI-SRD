""" List of all defaut Variables used in the tutorials present in `<https://github.com/pajjaecat/ORI-SRD/tree/main/Ressources/Notebooks>>`_ .

This module holds a list of all the variables used in the ORI package.




"""

attr_list = [('bus', 'name'),
             ('load', 'bus'),
             ('switch', 'bus'),
             ('line', 'from_bus'),
             ('line', 'to_bus'),
             ('trafo', 'hv_bus'),
             ('trafo', 'lv_bus')]
""" attr_list: list , Atribute list """

network_folder = '../Pickle_files/'
""" network_folder : str 

relative location of the network to study.

"""

excel_folder = '../Excel_files/'
""" excel_folder : str 

relative location of the network to study input's.

"""

py_folder = '../Modules/'
""" py_folder : str 

relative location of the used modules.

Notes
-----
    This file MUST be present in the ``py_folder``

"""

Δt = 1/6  # Time frequency 10mn ==> 1Hour/6
""" Δt : int, Time frequency"""


defAuth_hvBus_vRiseMax = 1.025
"""defAuth_hvBus_vRiseMax : float, Default max authorised voltage rise on Hv Buses on Lower network"""

defAuth_hvBus_vRiseMin = 0.95
"""defAuth_hvBus_vRiseMin : float, Default min authorised voltage rise on Hv Buses on Lower network"""

defAuth_lvBus_vRiseMax = 1.075  
"""defAuth_lvBus_vRiseMax : float, Default max authorised voltage rise on Lv Buses on Lower network"""

defAuth_lvBus_vRiseMin = 0.95
"""defAuth_lvBus_vRiseMin : float, Default min authorised voltage rise on Lv Buses on Lower network"""


default_hv_voltage = 20.6      
"""default_hv_voltage : float, Default = 20.6

Default voltage (kV) of Hv buses on Lower Network.

"""

default_lv_voltage = 0.4
"""default_lv_voltage : float, Default = 0.4

Default voltage (kV) of Lv buses on Lower Network.

"""

default_ctrld_hvProd_max = 4.0
"""default_ctrld_hvProd_max : float, Default = 4.

Default maximum output (MW ) of te controlled Hv producer

"""

h_start_end = ('07:00','18:50')
"""h_start_end : tuple, Default = ('07:00','18:50'), Daylight period"""


train_split_date = '2021 12 31 23:50' # Date of training+Validation split data Lower bond 
trainVal_split_date = '2021 06 01'     # lower date to split training and validation data
testSet_end_date = '2022 06 02'
testSet_start_date = '2021 06 03'



""" Module containing all the variables that will be used in the ORI problem """

# Create an attribute list to use in functions
attr_list = [('bus', 'name'),
             ('load', 'bus'),
             ('switch', 'bus'),
             ('line', 'from_bus'),
             ('line', 'to_bus'),
             ('trafo', 'hv_bus'),
             ('trafo', 'lv_bus')]

# Define a set of folders
network_folder = '../Pickle_files/'
excel_folder = '../Excel_files/'
py_folder = '../Modules/'

Δt = 1/6  # Time frequency 10mn ==> 1Hour/6

# Default Authorised voltage rise on hv Bus on lower Network
defAuth_hvBus_vRiseMax = 1.025     
defAuth_hvBus_vRiseMin = 0.95
defAuth_lvBus_vRiseMax = 1.075  
defAuth_lvBus_vRiseMin = 0.95

# Default voltage on hv and Lv Buses on lower Network (in KV)
default_hv_voltage = 20.6      
default_lv_voltage = 0.4

# Default max output of the controlled HV prod (MW)
default_ctrld_hvProd_max = 4.0

# Daylight period to consider
h_start_end = ('07:00','18:50')


train_split_date = '2021 12 31 23:50' # Date of training+Validation split data Lower bond 
trainVal_split_date = '2021 06 01'     # lower date to split training and validation data
testSet_end_date = '2022 06 02'
testSet_start_date = '2021 06 03'



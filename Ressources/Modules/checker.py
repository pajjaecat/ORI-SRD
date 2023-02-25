# -*- coding: utf-8 -*-

# Copyright (c) 2022-2023 by Jesse-James PRINCE AGBODJAN for SRD-Energies (https://www.srd-energies.fr/) 
# All rights reserved.

""" Modules where all the checking functions are defined """

import pandas


def check_input_concordance(self):
    """ Check if the inputs (models_folder_location, and models_name) have the same length 
    and raise an exception otherwise """
    if len(self._models_folder_location) != len(self._models_name):
        raise Exception('The inputs must have the same size')


def check_numberOf_plk_Files_in_folders(self):
    """ Check if the numbers of plk files in each folder given as argument is the same
    and raise an exception otherwise """

    # Create a new varible for simplicity call
    working_dict = self._sortedPlkFiles_in_folder_list_dict

    # Get the number of files in each sorted list
    max_files = [len(working_dict[elm]) for elm in working_dict]

    # List comprehension: 'equal' when the previous value is the same as the following
    # and 'notEqual' otherwise
    # egg:
    #      max_file = [20, 20 ,21 ], equal_list=['equal', 'notEqual']
    equal_list = ['equal' if (cur_elm == max_files[prev_index])
                  else 'notEqual' for prev_index, cur_elm in enumerate(max_files[1:])]

    if 'notEqual' in equal_list:
        raise Exception('The Numbers of plk files in folders are not the same.')
    # TODO: Give more indication in the output especially Give the number of the file
    # not present and the folder associated


def check_network_order(self):
    """ Make sure the upper network is bigger than the lower one"""
    if len(self._upperNet.bus) < len(self._lowerNet.bus):
        raise Exception('The first input (upper network) has a lower number of bus'
                        'compared to the second input (lower network). ')


def check_boxplot_inputs(self, ht_in, bt_in):
    """ Check if the inputs ht_in and bt_in are in the list of authorised value and
        raise and exception otherwise  """
    ht_authorised_values = list(self._P0100_max_range) + ['All']
    if ht_in not in ht_authorised_values:
        raise Exception(""" The ht argument input is not authorised. Make sure the argument is 
                            in arange(0., 4., 0.2) or is 'All'""")
    if bt_in not in self._bt_add_range:
        raise Exception(""" The bt argument input is not authorised. Make sure the argument is
                            in arange(0., 4., 0.2)""")


def check_countDictAsDf_input_inLocalSpace(self, input_var):
    """ Check if input_var that must be either of (1)self._vrise_count_dict or
    (2)self._capping_count_dict is already present in the local space of the instance,
    i.e. vRise_boxplot(*args) has already been executed once
    """
    try:
        getattr(self, input_var)
    except AttributeError:
        raise Exception("""The method *.vRise_boxplot(*args) must be run before 
                        calling .*countplot(*args)""")


def check_countplot_inputs(self, dict_name):
    """ Check if the input given by the user for countplot(*args) is authorised """
    if dict_name not in self._dict_vars:
        raise Exception(f' The input must be either of {list(self._dict_vars.keys())} ')


def check_coef_add_bt_dist(coef_add_bt_dist, includeNone=True):
    """ Check if the input coef_add_bt_dist is among the authorised values """
    authorised_list = [None, 'uppNet', 'lowNet', 'lowNet_rand']

    if not includeNone:
        authorised_list = ['uppNet', 'lowNet', 'lowNet_rand']

    if coef_add_bt_dist not in authorised_list:
        raise Exception(f' The parameter \'coef_add_bt_dist\' must be either of {authorised_list}')


def check_coef_add_bt_and_dist(self):
    """ check the concordance between coef_add_bt_and coef_add_bt_dist """
    if type(self._coef_add_bt) == type(self._coef_add_bt_dist):
        if type(self._coef_add_bt) in [int, float]:
            raise Exception(' Wrong type of parameter \'_coef_add_bt\' or \'_coef_add_bt_dist\'')
    elif type(self._coef_add_bt) is int:
        raise Exception(' The parameter \'coef_add_bt\' must be a float')
    elif type(self._coef_add_bt) is float:
        check_coef_add_bt_dist(self._coef_add_bt_dist, includeNone=False)


def check_hvProdName_in_lowerNet(self, controlled_hvProdName):
    """ Check if the name of the controlled HV producer exist in the lower network"""

    # Extract name of all HV prod in lower Net
    lowerNet_hvProdName_list = list(self._lowerNet.sgen.name[
                                        self._lowerNet.sgen.name.notna()]
                                    )
    if controlled_hvProdName not in lowerNet_hvProdName_list:
        raise Exception(f' Higher voltage producer {controlled_hvProdName} is not defined in the '
                        f'lower network. Existing producers are {lowerNet_hvProdName_list}')


def check_var_concordance(opf_status=False, pred_model=None):
    """
    Check the congruence between the optimal power flow variable and the type of prediction model.

    Parameters
    ----------
    opf_status: bool, optional, default = False
        Whether the maximum voltage is computed after a normal or optimal power flow or both
        + Normal  =>  **pandapower.runpp(net)**,  ofp_status = False
        + Optimal =>  **pandapower.runopp(net)**, ofp_status = True
        + Both    =>  A normal power flow is run. Only when the result i.e. max_vm_pu > threshold,
                      is the optimal power flow run.
    pred_model: String
        Which kind of prediction model to use for the all the variables to predict at current period
        + Pers  =>  Persistence model i.e. val(k)= val(k-1)

    """

    pred_model_values = ['Pers']

    # If the prediction model <pred_mod> is defined,  make sure that the <ofp_status> ='Both'
    if pred_model is not None:
        if pred_model not in pred_model_values:  # check if the pred_model value is an authorised
            raise ValueError('<pred_mod> must be either of', pred_model_values)

        if opf_status != 'Both':  #
            raise ValueError('Given that <pred_mod> is defined, <ofp_status>  must be  set to <\'Both\'> ')


def check_networkDataDf_columnsOrder(netInput_data_df: pandas.DataFrame):
    """  Check if the input dataframe column are in the expected order.

    The columns order **MUST** be  ['Cons', 'Prod_BT', 'P0_n', ...., 'P0_z'] \;
    where
        'Cons'    ==> The total demand, load or consumption on the upper network;
        'Prod_BT' ==> The total production of all lower voltage producers  on the upper network;
        'P0_n'    ==> The name of the first higher voltage producer on the lower Network;
        ...       ==> The name of the other Higher voltage producer on the lower network;
        'P0_z'    ==> The name of the last Higher voltage producer on the lower network


    """
    if netInput_data_df.columns[0] != 'Cons':
        raise Exception(f'The first column of the input dataframe MUST be the consumption or load and NAMED \'Cons\' ')

    if netInput_data_df.columns[1] != 'Prod_BT':
        raise Exception(f'The second column of the input dataframe MUST be the total production of all lower voltage '
                        f' producers in the upper network and must be NAMED or load and NAMED \'Prod_BT\' ')

    if not netInput_data_df.columns[2:].str.startswith('P').all():
        raise Exception(f'All of the higher voltage producers name MUST start by \'P\' ')


def check_resTables_existing(self):
    """ Check if the result tables are present in the _lowerNet pandapower table.

    Returns
    -------
    True
        Meaning the lowerNet is already run once using power flow, hence the res_bus and res_sgen
        tables exists and have a length not equal to 0
    False
        Results tables have length 0

    """
    return (len(self._lowerNet.res_bus) != 0) & (len(self._lowerNet.res_sgen) != 0)


def check_opf_status(opf_status: bool or str):
    """  Check whether the ``opf_status`` is authorised. """
    if opf_status not in ['Both', False]:
        raise ValueError('``opf_status`` must be either of [False, ''Both'']')


def check_clean(clean: bool):
    """ Check whether ``clean`` is authorised"""
    if type(clean) is not bool:
        raise ValueError('``clean`` must be a ``bool``')


def check_parResultsNames(par_result_name: str):
    """ Check whether ``par_result_name`` is a string """
    if type(par_result_name) is not str:
        raise TypeError(' ``gather_results()`` can only handle string as argument')


def check_robustnessParams(df_out_block_pfOpf, combRnn_param):
    """Check the concordance between ``df_out_block_pfOpf`` and ``combRnn_param`` """

    # When ``df_out_block_pf_opf`` has only one column, it implies that it is the first
    # output of :py:func:`oriFunctions.combineRnnPred`. In this case the last input of
    # :py:func:oriFunctions.robustness must be included.

    if (df_out_block_pfOpf.shape[1] == 1
            and (combRnn_param[0] is None
                 or combRnn_param[1] is None)):
        raise Exception(f'Since the input ``df_out_block_pfOpf`` has only one column, i.e. is'
                        f' the first output of :py:func:`oriFunctions.combineRnnPred`, the input'
                        f' ``combRnn_param``  **MUST** also be included.')

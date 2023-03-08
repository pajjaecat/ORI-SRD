# -*- coding: utf-8 -*-

# Copyright (c) 2022-2023 by Jesse-James PRINCE AGBODJAN for SRD-Energies (https://www.srd-energies.fr/) 
# All rights reserved.


"""
OriFunctions - Python library with List of all functions used in the
`tutorials <https://github.com/pajjaecat/ORI-SRD/tree/main/Ressources/Notebooks>`_

"""

import ipyparallel
import numpy
import oriClass
import pandapower
import pandas
from tqdm import tqdm  # Profiling
import time
import checker

# Import all variables from module oriVariables
from oriVariables import (network_folder,
                          excel_folder,
                          attr_list,
                          defAuth_hvBus_vRiseMax,
                          defAuth_hvBus_vRiseMin,
                          defAuth_lvBus_vRiseMax,
                          defAuth_lvBus_vRiseMin,
                          default_ctrld_hvProd_max,
                          default_hv_voltage,
                          default_lv_voltage
                          )

pd = pandas
np = numpy
pp = pandapower
ipp = ipyparallel


# ------------------------------------------------------------------------------------

##############################         FUNCTIONS          #########################################

def readAndReshape_excelFile(file_name: str,
                             folder_name: str = excel_folder,
                             n_row2read: int = None
                             ) -> numpy.array:
    """ Read and reshape in one dimension an excel file

    Read and reshape in a one dimension array the excel file given by file_name.

    Parameters
    ----------
    file_name : str
        Name of the file to load (with the correct extension).
    folder_name : str, optional, default = :py:data:`oriVariables.network_folder`
        Location of the folder where the file ``file_name`` is present.
    n_row2read : int, optional, default=0
         Numbers of rows to read in the excel file.


    """

    filename = f"{folder_name}{file_name}"
    cols_to_read = range(2, 8)  # Define index of columns to read
    # 0 10 20 30 40 50
    # Where each of the six columns to read represent a period.
    input_data = pandas.read_csv(filename,
                                 header=None,
                                 sep=";",
                                 usecols=cols_to_read,
                                 nrows=n_row2read)

    return numpy.array(input_data).reshape(-1) / 1000  # /1000 To convert data (MW)


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

def check_bus_connection(network,
                         bus_number,
                         attr_list_in: dict = attr_list
                         ):
    """

    Check and print the connection between a bus number and all the elements in the
    lower network.

    Parameters
    ----------
    network: pandapower network
        The network that has to be investigated.
    bus_number: list of int
        The number of the concerned bus(ses)
    attr_list_in: list of str tuple
        Each tuple in the list represents the attribute of the attribute
        to look for.
        Ex: `attr_list_in[0]` = ('bus', 'name') ==> network.bus.name
        must be accessed

    """

    for cur_bus in bus_number:  # For each bus
        for attribute in attr_list_in:  # For each tuple in the attribute list
            netsub = getattr(network, attribute[0])
            netsub_sub = getattr(netsub, attribute[1])

            if len(netsub[netsub_sub == cur_bus]) != 0:  # If there is some elements
                print(
                    f'----------******   Bus {cur_bus} net.{attribute[0]}.{attribute[1]} ******-------')
                print(netsub[netsub_sub == cur_bus], '\n')
        print('\n')


# ____________________________________________________________________________________________________________--
# _____________________________________________________________________________________________________________

def run_powerflow(network,
                  lowNet_hv_activBus: list,
                  sum_max_p_mw_upperNet: tuple,
                  dict_df_sgenLoad: dict,
                  opf_status=False
                  ):
    """ Initialise the parameters of the network

    Return a list of maximum voltage on the network for each period given
    by the index of element in.

    Parameters
    ----------
    network : Pandapower network
        The lower network to be investigated.
    lowNet_hv_activBus : list
        list of all Hv buses activated in `network`.
    sum_max_p_mw_upperNet: tuple
        Sum of maximum power seen from the upper network (in the tutorials,
        saint laurent  compared to the lower network civaux)
            ``sum_max_p_mw_upperNet[0]`` :
                Of all Lower voltage producers.
            ``sum_max_p_mw_upperNet[1]`` :
                Of all Load in the upper network.
    dict_df_sgenLoad : dict
        Dictionary of dataframe. For the first three keys, the corresponding df must
        indexed by the periods of the considered year. The keys must be the
        following:
            `df_prodHT` :  pandas.DataFrame
                Power of all Higher voltage producers in the lower network. The
                columns represent each Hv producer.
            `df_prod_bt_total` :  pandas.DataFrame
                Total power of all lower voltage producers seen  from  the upper
                Network.
            `df_cons_total` :  pandas.DataFrame
                Total Load demand seen from the upper Network.
            `lowerNet_sgenDf_copy` :  pandas.DataFrame
                Copy of all the static generator (hv & lV) in the lower network.
    opf_status: bool or str, optional, default=False
        Optimal power flow status. Whether the maximum voltage rise on the lower
        network
        HV buses is extracted after a power flow, an optimal power flow  or both.
        Three values are possible:
            ``opf_status`` = False :
                Run a simple power flow i.e., `pandapower.runpp(network)`
            ``opf_status`` = True :
                Run an optimal power flow i.e., `pandapower.runopp(network)`
            ``opf_status`` = "Both" :
                A power flow is run. Only when the result i.e. the voltage rise
                detected on hv Prod Buses ``max_vm_pu`` >
                :py:data:`oriVariables.defAuth_hvBus_vRiseMax`, is the  optimal
                power flow run.

    Returns
    -------
    tuple or list
        The return value depends on ``opf_status``

    See Also
    --------
    run_powerflow_at : Run power flow at a specific period

    Notes
    -----
    It is not recommended to use  this function since its implementation is not
    optimal for
    use with parallel engines created with `ipyparallel <https://ipyparallel.readthedocs.io/en/latest/>`_ .
    We recommend using instead :py:func:`oriFunctions.run_powerflow_at` .

    """

    # Creating empty list
    list_max_vm_pu = []  # Maximum vm_pu at each period considered
    list_sgen_HT = []  # Actual HT generators power after optimal flow

    # Initiate parameters from inputs
    df_prodHT = dict_df_sgenLoad['df_prodHT']

    # Initialise the network and get the maximum value for each period
    for cur_period in tqdm(df_prodHT.index):
        if opf_status:  # Run optimal power flow
            max_vm_pu, (hvProd_afterOPF, lvProd_afterOPF) = run_powerflow_at(network, cur_period,
                                                                             lowNet_hv_activBus,
                                                                             sum_max_p_mw_upperNet,
                                                                             dict_df_sgenLoad,
                                                                             opf_status)
            list_max_vm_pu.append(max_vm_pu)
            list_sgen_HT.append(hvProd_afterOPF)

        else:  # Run simple power flow
            hvProd_pf, lvProd_pf = run_powerflow_at(network,
                                                    cur_period,
                                                    lowNet_hv_activBus,
                                                    sum_max_p_mw_upperNet,
                                                    dict_df_sgenLoad,
                                                    opf_status)

            list_max_vm_pu.append(hvProd_pf)

    # Return depends on opf_status
    if opf_status:
        return list_max_vm_pu, list_sgen_HT
    else:
        return list_max_vm_pu


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

def run_powerflow_at(network,
                     cur_period,
                     lowNet_hv_activBus: list,
                     sum_max_p_mw_upperNet: tuple,
                     dict_df_sgenLoad: dict,
                     auth_max_VriseHvBus: float = defAuth_hvBus_vRiseMax,
                     opf_status: (bool or str) = False,
                     pred_model: str = None
                     ):
    """Run PF or OPF at ``cur_period`` depending on ``opf_status``.

    Parameters
    ----------
    network : Pandapower network
        The lower network to consider.
    cur_period : str or pandas.Period
        The period at which the pf/opf must be run.
    lowNet_hv_activBus : list
        list of all Hv buses activated in the concerned network.
    sum_max_p_mw_upperNet : tuple
        Sum of maximum power seen from the upper network (here, saint laurent
        compared to the lower network civaux).
            ``sum_max_p_mw_upperNet[0]`` :
                Of all Lower voltage producers.
            ``sum_max_p_mw_upperNet[1]`` :
                Of all Load in the upper network.
    dict_df_sgenLoad : dict
        Dictionary of dataframe. For the first three keys, the corresponding df must
        indexed by the periods of the considered year. The keys must be the
        following:
            `df_prodHT` : pandas.DataFrame
                Power of all Higher voltage producers in the lower network. The
                columns represent each Hv producer.
            `df_prod_bt_total` : pandas.DataFrame
                Total power of all lower voltage producers seen  from  the upper
                Network.
            `df_cons_total` : pandas.DataFrame
                Total Load demand seen from the upper Network.
            `lowerNet_sgenDf_copy` : pandas.DataFrame
                Copy of all the static generator (hv & lV) in the lower network.
    auth_max_VriseHvBus: float, optional default = :py:data: `oriVariables.defAuth_hvBus_vRiseMax`
        Threshold of maximum voltage allowed on the HV buses of `network`. This
        parameter is used only when ``opf_status`` = `"Both"`
    opf_status: bool or str, optional, default=False
        Optimal power flow status. Whether the maximum voltage rise on the lower
        network HV buses is extracted after a power flow, an optimal power flow
        or both.  Three values are possible:
            ``opf_status`` = False :
                Run a simple power flow i.e., `pandapower.runpp(network)`
            ``opf_status`` = True :
                Run an optimal power flow i.e., `pandapower.runopp(network)`
            ``opf_status`` = "Both" :
                A power flow is run. Only when the result i.e. the voltage rise
                detected on hv Prod Buses ``max_vm_pu`` > :py:data:`oriVariables.defAuth_hvBus_vRiseMax`,
                is the  optimal power flow run.
    pred_model: str, optional, default = None
        Which kind of prediction model to use for the all the variables to
        predict at current period.
            None :
                No prediction model is used
            "Pers" :
                Persistence model i.e. val(k)= val(k-1)

    Returns
    -------
    Depends on ``opf_status``
        ``opf_status`` = False
            max_vm_pu_pf, cur_period
        ``opf_status`` = True
            cur_max_VriseHvBus, (hvProd_afterOPF, lvProd_afterOPF), cur_period
        ``opf_status`` = "Both"
            [max_vm_pu_pf, cur_max_VriseHvBus],(hvProd_afterOPF, lvProd_afterOPF), cur_period \n
            where
                max_vm_pu_pf : float
                    Maximum voltage given by the power flow
                cur_max_VriseHvBus : float
                    Maximum voltage rise detected on all the Hv buses on the lower
                    Network, given by OPF.
                hvProd_afterOPF : list
                    List (in the order  in which they appear in the pandapower
                    network sgen table) of the optimal power that each hv producer
                    on the lower net must inject in order  to satisfy  the
                    `auth_max_VriseHvBus`.
                lvProd_afterOPF: list
                    List (in the order  in which they appear in the pandapower
                    network sgen table) of the optimal power that each lv producer
                    on the lower net must inject in  order  to satisfy the
                    `auth_max_VriseHvBus`.
                cur_period : pandas.Period
                    The period at which the PF/OPF is run.

    Raises
    ------
    ValueErrorException
        If ``opf_status`` is not in [True, False, "Both"]

    Notes
    -----
    For the moment ``auth_max_VriselvBus`` i.e. the authorised voltage rise on the
    lv buses constraint is considered only when the Voltage rise on the hv buses
    is greater than  ``auth_max_VriseHvBus``. Simply put, as long as no voltage
    rise above ``auth_max_VriseHvBus`` is detected one does not care about the
    value of the voltage rise on the lv buses.
    TODO : Considered the auth_max_VriselvBus to run an opf.

    """

    # Check variables congruence
    checker.check_var_concordance(opf_status, pred_model)

    # -- GT1
    if pred_model == 'Pers':  # if the the prediction model is the persistence,
        cur_period = cur_period - 1

    # Initialize the network. See the corresponding function for more explanation
    initLowerNet_at(network, cur_period, sum_max_p_mw_upperNet, dict_df_sgenLoad)

    # Get the maximum voltage magnitude of all activated bus to a list. See the
    #                                corresponding function for more explanation
    if opf_status is True:  # Run optimal power flow ******************************************

        # get maximum value of vm_pu for the current period after optimal power flow
        cur_max_VriseHvBus = max_vm_pu_at(network, cur_period,
                                          lowNet_hv_activBus,
                                          dict_df_sgenLoad, opf_status)

        # Get the value of HT producer after optimal flow.
        hvProd_afterOPF = list(network.res_sgen[network.sgen.name.notna()].p_mw)
        lvProd_afterOPF = list(network.res_sgen[network.sgen.name.isna()].p_mw)

        # Depending on the prediction model parameter the return is different ----------------
        # For <pred_model = 'Pers'> given that at GT1 the <cur_period = cur_period-1> one must
        #  reset cur_period to its initial value using <cur_period+1> before returning the results
        if pred_model == 'Pers':
            return cur_max_VriseHvBus, (hvProd_afterOPF, lvProd_afterOPF), cur_period + 1
        else:
            return cur_max_VriseHvBus, (hvProd_afterOPF, lvProd_afterOPF), cur_period

    elif opf_status == 'Both':  # Run normal and depending on the situation, also optimal power flow  **
        # run power flow first
        cur_max_VriseHvBus = max_vm_pu_at(network, cur_period, lowNet_hv_activBus,
                                          dict_df_sgenLoad, False)
        max_vm_pu_pf = cur_max_VriseHvBus  # Save the maximum voltage given by the power flow
        # before optimizing
        # If the maximum voltage on buses is above the authorised threshold, run optimal power flow
        if cur_max_VriseHvBus > auth_max_VriseHvBus:
            cur_max_VriseHvBus = max_vm_pu_at(network, cur_period, lowNet_hv_activBus,
                                              dict_df_sgenLoad, True)

        # Get the value of HV and LV producers after optimal flow.
        hvProd_afterOPF = list(network.res_sgen[network.sgen.name.notna()].p_mw)
        lvProd_afterOPF = list(network.res_sgen[network.sgen.name.isna()].p_mw)

        # Depending on the prediction model parameter the return is different ----------------
        # For <pred_model = 'Pers'> given that at GT1 the <cur_period = cur_period-1> one must
        #  reset cur_period to its initial value using <cur_period+1> before returning the results
        if pred_model == 'Pers':
            return [max_vm_pu_pf, cur_max_VriseHvBus], (hvProd_afterOPF, lvProd_afterOPF), cur_period + 1
        else:
            return [max_vm_pu_pf, cur_max_VriseHvBus], (hvProd_afterOPF, lvProd_afterOPF), cur_period

    elif opf_status is False:  # Run normal power flow  ***************************************************
        return max_vm_pu_at(network, cur_period, lowNet_hv_activBus, dict_df_sgenLoad, opf_status), cur_period

    else:
        raise ValueError('<opf_status> must be either of [True, False, "Both"]')


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________


def initLowerNet_at(network,
                    cur_period,
                    sum_max_p_mw_upperNet: tuple,
                    dict_df_sgenLoad: dict
                    ):
    """

    Initialise the parameters of the network at the `cur_period`.

    Parameters
    ----------
    network : Pandapower network
        The lower level network to initialize.
    cur_period : pandas.Period
        The period at which ``network`` must be initialized.
    sum_max_p_mw_upperNet : tuple
        Sum of maximum power seen from the upper network (here, saint laurent
        compared to the lower network civaux).
            ``sum_max_p_mw_upperNet[0]`` :
                Of all Lower voltage producers.
            ``sum_max_p_mw_upperNet[1]`` :
                Of all Load in the upper network.
    dict_df_sgenLoad : dict
        Dictionary of dataframe. For the first three keys, the corresponding df
        must indexed by the periods of the considered year. The keys must be the
        following:
            `df_prodHT` :  pandas.DataFrame
                Power of all Higher voltage producers in the lower network. The
                columns represent each Hv producer.
            `df_prod_bt_total` :  pandas.DataFrame
                Total power of all lower voltage producers seen  from  the upper
                Network.
            `df_cons_total` :  pandas.DataFrame
                Total Load demand seen from the upper Network.
            `lowerNet_sgenDf_copy` :  pandas.DataFrame
                Copy of all the static generator (hv & lV) in the lower network.

    Notes
    -----
    The following parameters are initialized
        #TODO ?? ADD the Initialized Parameters#


    """
    ##  TODO : Give only the data of the current period to
    ##  the function instead of that of the whole year

    # Initiate parameters to be used within function
    upNet_sum_max_lvProd = sum_max_p_mw_upperNet[0]
    upNet_sum_max_load = sum_max_p_mw_upperNet[1]

    df_prodHT = dict_df_sgenLoad['df_prodHT']
    df_prod_bt_total = dict_df_sgenLoad['df_prod_bt_total']
    df_cons_total = dict_df_sgenLoad['df_cons_total']
    df_lowNetSgen_cp = dict_df_sgenLoad['lowerNet_sgenDf_copy']

    # Initialise HT producers
    network.sgen.p_mw[network.sgen.name.notna()] = df_prodHT.loc[cur_period].values

    # Mask Lower voltage producers
    mask_lvProd = df_lowNetSgen_cp.name.isna()

    # Initialized maximum power of LV producers
    # Why reinitialize the max power?  Because in the function max_vm_pu_at(*args), the maximum power
    # is set to the actual production given opf  constraints .
    network.sgen.loc[mask_lvProd, 'max_p_mw'] = df_lowNetSgen_cp[mask_lvProd].max_p_mw

    # Initialize real output of LV producers
    prod_bt_total_1mw = df_prod_bt_total.loc[cur_period].values / upNet_sum_max_lvProd
    network.sgen.p_mw[network.sgen.name.isna()] = (network.sgen.max_p_mw[network.sgen.name.isna()]
                                                   * prod_bt_total_1mw
                                                   )
    # Initialize Loads
    load_total_1mw = df_cons_total.loc[cur_period].values / upNet_sum_max_load
    network.load.p_mw = (network.load.max_p_mw * load_total_1mw)

    # Work with julia Power model since the load is zero
    # network.load.p_mw = (network.load.max_p_mw*df_cons_total.loc[cur_period].
    # values*0/upNet_sum_max_load)


# _____________________________________________________________________________________________________________--
# _____________________________________________________________________________________________________________

def max_vm_pu_at(network,
                 cur_period,
                 lowNet_hv_activBus: list,
                 dict_df_sgenLoad: dict,
                 opf_status: (bool or str) = False):
    """

    Extract the maximum voltage over all the higher voltages active buses in the
    network at the current period.

    Parameters
    ----------
    network : Pandapower network
        The network to consider
    cur_period : str or pandas.Period
        The period to investigate.
    lowNet_hv_activBus : List
        List of all the higher voltage activated bus in the lower network
    dict_df_sgenLoad : dict
        Dictionary of dataframe. For the first three keys, the corresponding df
        must indexed by the periods of the considered year. The keys must be the
        following:
            `df_prodHT` : pandas.DataFrame
                Power of all Higher voltage producers in the lower network. The
                columns represent each Hv producer.
            `df_prod_bt_total` : pandas.DataFrame
                Total power of all lower voltage producers seen  from  the upper
                Network.
            `df_cons_total` : pandas.DataFrame
                Total Load demand seen from the upper Network.
            `lowerNet_sgenDf_copy` : pandas.DataFrame
                Copy of all the static generator (hv & lV) in the lower network.
    opf_status: bool or str, optional, default=False
        Optimal power flow status. Whether the maximum voltage rise on the lower
        network HV buses is extracted after a power flow, an optimal power flow  or
        both. Three values are possible:
            ``opf_status`` = False :
                Run a simple power flow i.e., `pandapower.runpp(network)`
            ``opf_status`` = True :
                Run an optimal power flow i.e., `pandapower.runopp(network)`
            ``opf_status`` = "Both" :
                A power flow is run. Only when the result i.e. the voltage rise
                detected on hv Prod Buses ``max_vm_pu`` > ``auth_max_VriseHvBus``,
                is the  optimal  power flow run.

    Returns
    -------
    max_vm_pu :
        Maximum voltage rise over all the HV buses in the lower network at the
        ``cur_period``.

    Notes
    -----
    TODO : Return the maximum voltage rise over all the LV buses in the lower network
    for the current instant. In this case one needs to add as input to the function
    the ``net_lv_activBus`` list as  well. Hence one can replace the
    ``lowNet_hv_activBus`` by a tuple of ``(lowNet_hv_activBus, uppNet_lv_activBus)``.


    """

    # Initiate parameters from input
    df_prodHT = dict_df_sgenLoad['df_prodHT']
    df_lowNetSgen_cp = dict_df_sgenLoad['lowerNet_sgenDf_copy']

    if opf_status:  # If status is True or 'Both'

        # Create a mask of all controlled LV producer in the lower Network
        mask_ctrld_lvProd = df_lowNetSgen_cp.name.isna() & df_lowNetSgen_cp.controllable

        # Extract the name and the index of the controlled Higher voltage producer.
        # This supposed the there is a controllable column in the network, this controllable column
        # is true for the controlled HV producer
        hvSgen_df = network.sgen[network.sgen.name.notna()]
        ctrld_hvProd_name = list(hvSgen_df[hvSgen_df.controllable].name)[0]
        ctrld_hvProd_ind = list(hvSgen_df[hvSgen_df.controllable].index)[0]

        # update
        # For optimal flow, given that the sgen P0100 is controllable the optimization
        # result is to draw the maximum power  with no regard to the actual power provided
        # at each instant. To alleviate this problem we would rather initialize the maximum
        # power of the said  producer with the actual production.
        network.sgen.at[ctrld_hvProd_ind, 'max_p_mw'] = df_prodHT[ctrld_hvProd_name][cur_period]

        # Same process for the controlled LV producers as for the controlled HV Sgen
        # Note that network.sgen.p_mw has already been initialized in the upper function
        # initLowerNet_at(*args)
        network.sgen.loc[mask_ctrld_lvProd, 'max_p_mw'] = network.sgen.p_mw[mask_ctrld_lvProd]

        pandapower.runopp(network, delta=1e-16)  # Run network
        # Due to convergence problems, I've decreased the power tolerance as done in
        # [https://github.com/e2nIEE/pandapower/blob/v2.11.1/tutorials/opf_basic.ipynb] and changed
        # the initial starting solution of opf

        # pandapower.runpm_ac_opf(network) # Run network with Julia Power model:
        # Not converging for the moment, but Do converge when le load demand is low

    else:
        pandapower.runpp(network)  # Run network
        # pandapower.runpm_pf(network) # Run network with Julia Power model:
        # Not converging for the moment, but Do converge when le load demand is low

    # Return the maximum voltage rise over all the HV buses in the lower network for the current instant
    return network.res_bus.loc[lowNet_hv_activBus, 'vm_pu'].max()
    # TODO : Return the maximum voltage rise over all the LV buses in the lower network for the current
    #        instant. In this case one needs to add as input to the function the  net_lv_activBus
    #        list as well. Hence one can replace the lowNet_hv_activBus by a tuple
    #        of (lowNet_hv_activBus, uppNet_lv_activBus).


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

def improve_persinstence(per_extracted_res_df,
                         df_prodHT,
                         auth_max_VriseHvBus: float = defAuth_hvBus_vRiseMax,
                         h_start_end: list = ['11:00', '14:00']):
    # Implement : * Inject all the production as long as max_vm_pu_pf < vm_mu_max, i.e.
    # no voltage rise is detected
    """     Improve the results given by the persistence model.


    If a voltage rise is not predicted by the persistence model at a certain period,
    the controllable sgens is allowed to inject all its power into the grid.
    Otherwise the energy producer can inject at most the predicted power by the
    persistence model.


    .. deprecated:: 1.0.1
          This function is deprecated. Will be removed in the next version.

    Parameters
    ----------
    per_extracted_res_df :  pandas.DataFrame
        Result given by the persistence model. Output of :py:func:`oriFunctions.extract_par_results`.
    df_prodHT :  pandas.DataFrame
        Dataframe containing data of all the HT producers in the network.
    auth_max_VriseHvBus : float, Optional, Default = :py:data:`oriVariables.defAuth_hvBus_vRiseMax`
        Threshold of maximum voltage allowed on the HV buses.
    h_start_end : list of str, optional, default = ['11:00', '14:00']
        Hour between which the HV can inject all is production into the network


    Returns
    -------
    per_improved_res :  pandas.DataFrame
        Output the improve persistence model improve by the previously described
        strategy.

    """

    # Copy the results of the persistence model
    per_improved_res_out = per_extracted_res_df.copy(deep=True)
    per_improved_res = per_extracted_res_df.copy(deep=True)

    # Convert index from period to timestamp
    per_improved_res.index = per_improved_res.index.to_timestamp()

    # Extract the part of the df one want to work with i.e. the period before h_start
    # and after h_end as -------'11:00'     '14:00'------ for the default value
    # the period defined between h_start and h_end is not considered since voltage rises
    # are known to happen in that interval
    working_df = per_improved_res.between_time(h_start_end[1], h_start_end[0])

    # Extract index of instances where no voltage rise is detected ignoring the last one
    # because not present in the initial df df_prodHT
    var_index = working_df[working_df.max_vm_pu_pf <= auth_max_VriseHvBus].index.to_period('10T')[:-1]

    # Replace the prediction from the persistence model with the actual production since
    # no voltage rise is detected at these periods
    per_improved_res_out.P0100[var_index] = df_prodHT.P0100[var_index]

    return per_improved_res_out, var_index


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

def prediction_bloc(rnn_model,
                    fitting_scaler,
                    history,
                    scaler_features=None
                    ):
    """

    Prediction bloc Using a RNN (LSTM).  Predict the values for the the next period.

    Parameters
    ----------
    rnn_model : Recurrent neural network
        The model that will be used to predict the value at the next period.
    fitting_scaler : Scaler
        Scaler parameters that are used to transform the training data set
        fed to the ``rnn_model``
    history :
        Non scaled history of the Electrical network.
    scaler_features : Scaler, optional, default = None
        Scaler to use for prediction when the number of
        variables to predict is different from the number of features

    Returns
    -------
    tuple of list
        var_predicted : list of float
            Prediction of the interest variables at ``pred_period``
        pred_period :  pandas.Period or str
            Period for which the prediction is done.

    """

    history_last_ind = history.index[-1]  # Get index of the last period of history
    in_shape = tuple([1]) + rnn_model.input_shape[1:]  # define input shape for the RNN

    # Scaled the input  based on the fitting scaler
    scaled_history = fitting_scaler.transform(history).reshape(in_shape)

    pred = rnn_model.predict(scaled_history, verbose=False)  # prediction

    # inverse transform the prediction
    if scaler_features is None:
        pred_inv_trans = fitting_scaler.inverse_transform(pred)
    else:
        pred_inv_trans = scaler_features.inverse_transform(pred)

    # Return the prediction of the RNN and the time period associated ()
    return pred_inv_trans, history_last_ind + 1


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

def predictionBin_bloc(rnn_model,
                       fitting_scaler,
                       history,
                       sig_thresh: float = 0.5
                       ):
    """ Binary prediction

    Binary prediction bloc Using a RNN (LSTM). Predict whether a voltage above the
    defined threshold will occur at the next period.

    Parameters
    ----------
    rnn_model : Recurrent neural network;
        The binary trained Recurrent Neural network
    fitting_scaler : Scaler
        Scaler parameters that are used to transform the training data set
        fed to the `rnn_model`
    history :
        Non scaled history of the Electrical network:
    sig_thresh : float, optional, default = 0.5
        Value used to threshold `rnn_model` output.


    Returns
    -------
    tuple of list
        var_predicted : {0, 1}
            Prediction of voltage rise above the defined thresholds. ``1`` and ``0``
            predicted implies respectively that the defined threshold is and is not
            respected.
        pred_periods :  pandas.Period or str
            Period for  which the prediction is done.

    """

    history_last_ind = history.index[-1]  # Get index of the last period of history
    in_shape = tuple([1]) + rnn_model.input_shape[1:]  # define input shape for the RNN

    # Scaled the input  based on the fitting scaler
    scaled_history = fitting_scaler.transform(history).reshape(in_shape)

    pred = rnn_model.predict(scaled_history, verbose=False)  # prediction

    pred_bin = (pred > sig_thresh).astype(int)  # convert prediction into a binary variablethe prediction

    # Return the prediction of the RNN and the time period associated ()
    return pred_bin[0][0], history_last_ind + 1


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

def combineRnnPred(model_Vrise_dict,
                   ctrlHvProd_model1_out,
                   auth_max_VriseHvBus: float = defAuth_hvBus_vRiseMax,
                   n_models=None):
    """ Combine the prediction of Three RNN to get a better prediction as an
    `Ensemble model <https://en.wikipedia.org/wiki/Ensemble_learning>`_.


    .. deprecated:: 1.0.1
        Will be removed in the next version. Developing a more robust
        approach using an RNN that combines directly all the three models during
        training.

    .. versionchanged:: 1.0.2
        Was previously name :py:func:`robustPred`


    Parameters
    ----------
    model_Vrise_dict : Dict
        Dictionary of the voltage rise for each model
    ctrlHvProd_model1_out :  pandas.DataFrame
        Partial output of :py:class:`oriClass.CreateParEngines.get_results_asDf`
        that is :math:`\\tilde{Y}(k)` the predicted value by model1 of the
        ctrlHvProd. This is the command value to send to the said Hv Prod when
        the function predicts an exceeding of   the defined threshold
        ``auth_max_VriseHvBus``.
    auth_max_VriseHvBus : float, optional default = :py:data:`oriVariables.defAuth_hvBus_vRiseMax`
        Threshold of maximum voltage allowed on the HV buses of `network`.
    n_models : int or str, optional, default=None
        Number of models which must agree on voltage rise above threshold before
        a command is set to the controlled HV producer.
            1 :
              At Least one of the models
            2 :
              At least two of the models
            3 :
              All three models
            "Modelx" :
                Name of the Model which voltage rise above threshold prediction is
                considered with x in {1,2,3}

    Returns
    -------
    new_HvCtrldProd_df :  pandas.DataFrame
        y_optimal after combined models.
    bin_thresh_df :  pandas.DataFrame

    Raises
    ------
    ValueErrorException
        If ``n_models`` is the wrong type or the wrong value.

    Notes
    -----
    In the future versions, the function **MUST** be recoded such that it follows
    exactly the block scheme  described is section 2.1 of
    `VRiseControlBlockScheme <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Docs/VRiseControlBlockScheme.pdf>`_.
    For the moment it has been implemented differently i.e. the block PF in Model 1
    computes both PF and OPF.
    To change it, compute solely PF in model 1 and based on the combination of the
    three models in the  block Comb, compute OPF solely for the interest periods. The
    interest periods are the periods where the  combination of Models agree on
    non-respect of the :py:data:`oriVariables.defAuth_hvBus_vRiseMax`.

    """

    # Extract model voltage rise from Input dictionary
    model1_Vrise, model2_Vrise, model3_Vrise = (model_Vrise_dict['Model1'],
                                                model_Vrise_dict['Model2'],
                                                model_Vrise_dict['Model3'])

    mask_per2work = model1_Vrise.index  # Get index of the considered period
    vect_int = np.vectorize(int)  # vectorized version of int

    # Get controlled HV prod Name
    ctrld_hvProdName = ctrlHvProd_model1_out.columns[0]

    # Create an empty dataframe i.e. binary threshold
    bin_thresh_df = pd.DataFrame(index=mask_per2work)

    # add the binary output of three models to the created df
    bin_thresh_df[['Model2']] = model3_Vrise.values
    bin_thresh_df[['Model3']] = vect_int(model2_Vrise > auth_max_VriseHvBus)
    bin_thresh_df[['Model1']] = vect_int(model1_Vrise > auth_max_VriseHvBus)

    # Combined_output of all models
    bin_thresh_df[['Model_All']] = np.array(bin_thresh_df.sum(axis=1)).reshape((-1, 1))

    # Create a new dataframe for the controlled SGEN based on its real values
    df_ctrlHvProd_toReturn = ctrlHvProd_model1_out.loc[mask_per2work, [ctrld_hvProdName]]

    if type(n_models) is str:  # If n_model is a string
        if n_models in model_Vrise_dict.keys():  # Check if the sting input is in the model Dict
            vrise_true_mask = bin_thresh_df[n_models] == 1  # Create the mask using only the
            # period where the concerned model predict there is an voltage rise
        else:
            raise ValueError('Since <n_models> is a string it must be be either of',
                             list(model_Vrise_dict.keys()))

    elif type(n_models) is int:  # If n_model is int
        if n_models <= 3:
            # Create mask of instants where at least n models agrees on voltage rise above threshold
            vrise_true_mask = bin_thresh_df.Model_All >= n_models
        else:
            raise ValueError('Since <n_models> is an int it must be defined such that 0 < n_models <= 3 ')

    else:
        raise ValueError('<n_models> is the wrong type. Must either be an int or a string')

    # Use vrise_true_mask to insert predicted values given by model1 at the concerned instants
    df_ctrlHvProd_toReturn[vrise_true_mask] = ctrlHvProd_model1_out.loc[mask_per2work[vrise_true_mask],
                                                                        [ctrld_hvProdName]]

    return df_ctrlHvProd_toReturn, bin_thresh_df


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

def extractParRes_asDf(parallel_result,
                       df_prodHT):
    """ Extract Parallel Engines Results.

    Extract and save the result of the parallel computation in a dataframe that is
    output

    Parameters
    ----------
    parallel_result : list
        Parallel engine results, i.e. ``dview.gather(*args)``
    df_prodHT :  pandas.DataFrame
        Dataframe of all the Higher voltage sgen in the lower network

    Returns
    -------
    pandas.DataFrame
        Parallel results extracted as a dataframe.

    Warnings
    --------
    Will be removed in the next version. Use :py:func:`oriClass.CreateParEngines.get_results_asDf`

    """

    # Get df_prodHT columns name [] from one of the engines
    df_prodHT_colName = df_prodHT.columns

    # Get all the elements from the parallel result in a list
    # elm[0]      : Maximum voltage on all the line
    # elm[1]      : Power injected into the network by all the Sgen in the network
    # elm[1][0][0]: Power injected into the network by the first HV producer
    # ...
    # elm[1][0][n]: Power injected into the network by the last HV producer i.e. P0100
    # elm[1][1][0]: Power injected into the network by the first LV producer
    # ...
    # elm[1][1][n]: Power injected into the network by the Lasr LV producer
    # elm[2]   : Period index associated to all the previous output variable

    # elm[0] can either be a list of [max_vm_pu_pf : max voltage  before opf
    #                                 max_vm_pu : maximum voltage after opf]
    # or a single float which is  max_vm_pu : maximum voltage after opf.
    # See the function run_powerflow_at (*args, opf_status='both', pred_model= 'Pers')

    SumLv_colName = ['SumLv']  # sum of the injected power of all lower voltage producers
    # in the network

    if type(parallel_result[0][0]) is list:
        sep_list = [(*elm[0], *elm[1][0], np.array(elm[1][1]).sum(), elm[2])
                    for elm in parallel_result]
        # Create a columns using 'vm_pu_max' and add the HT producers name
        colls = ['max_vm_pu_pf', 'max_vm_pu'] + df_prodHT_colName.to_list() + SumLv_colName
    else:
        sep_list = [(elm[0], *elm[1][0], np.array(elm[1][1]).sum(), elm[2])
                    for elm in parallel_result]
        # Create a columns using 'vm_pu_max' and add the HT producers name
        colls = ['max_vm_pu'] + df_prodHT_colName.to_list() + SumLv_colName

    # Create a data based on all the cols of sep_list except the last one that is the index
    data_input = np.array(np.array(sep_list)[:, :-1], dtype=float)
    index_list = np.array(sep_list)[:, -1]  # extract the last col that is the index

    # create new  dataframe based on previous unpack data
    df = pd.DataFrame(data=data_input, index=index_list, columns=colls)

    # return the newly create dataFrame with the index sorted
    return df.sort_index()


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

def _upscale_HvLv_prod(prod_hv2upscale_df,
                       prod_lv2upscale_df,
                       ctrld_hvProd_max,
                       upNet_sum_max_lvProd,
                       cur_hvProd_max: float = 0,
                       params_coef_add_bt: tuple = (None, None)
                       ):
    """

    Upscale  both the controlled Higher voltage(HV) producer (P0100) in the lower
    network (civeaux) and the total Lower voltage (LV) production. `coef_add_bt_dist`
    allow to choose how the upscaling is done on the LV production. This mean the BT
    producer on the lower network receives only a fraction of the added BT production.
    See function upscale_HvLv_prod() for the version of the function where the BT
    producer receive all the Added BT prod


    Parameters
    ----------
    prod_hv2upscale_df :  pandas.DataFrame
        dataframe of the HV prod to upscale i.e. P0100
    prod_lv2upscale_df :  pandas.DataFrame
        dataframe of the total LV producers output (That must be increased) i.e.
        Prod_BT
    ctrld_hvProd_max : float
        Maximum fixed output of the Controlled Higher voltage producer (MW)
    upNet_sum_max_lvProd  : float
        Sum of maximum output of all lower voltage (LV) producers (MW) in the upper
        Network.
        TODO** Get upNet_sum_max_lvProd from oriClass.InitNetwork() instance
    cur_hvProd_max : int, optional, default = 0
        Current value of maximum output Power of the HV producer (MW)
    params_coef_add_bt : tuple
        Parameters associated with how the upscaling of the total LV production is
        done. See doc :py:func:`oriClass.InitNetworks` for more information
            coef_add_bt :
            coef_add_bt_dist :

    Returns
    -------
    tuple
        The upscaled version of the HV producer and LV

    """

    ###  ----------- Upscale P0100 the controlable HT producer  ---------------  ###
    prodHT_P0100_1mw_df = prod_hv2upscale_df / ctrld_hvProd_max  # Rescale the total HT production for 1mw
    upscaled_prodHT_P0100_df = cur_hvProd_max * prodHT_P0100_1mw_df  # Redefine the HT production based on the rescale

    # Get the parameters coef_add_bt and coef_add_bt_dist
    coef_add_bt = params_coef_add_bt[0]
    coef_add_bt_dist = params_coef_add_bt[1]

    ###  ----------- upscale BT production ---------------  ###
    # Only the upscaling on the upper Network is done here. For the other cases,
    # i.e. 'lowNet' and 'lowNet_rand', see oriClass.InitNetworks(*args)
    upscaled_prod_bt_total_df = prod_lv2upscale_df
    if coef_add_bt_dist == 'uppNet':
        # Rescale the total BT production for 1mw
        prod_bt_total_1mw_df = prod_lv2upscale_df / upNet_sum_max_lvProd

        # Increase sum of output of BT  the coeef_add_bt if coef_add_bt_dist
        upscaled_prod_bt_total_df = prod_lv2upscale_df + coef_add_bt * prod_bt_total_1mw_df

    upscaled_prod_bt_total_df.columns = ['Prod_BT']

    return upscaled_prodHT_P0100_df, upscaled_prod_bt_total_df


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

def createDict_prodHtBt_Load(df_pred_in,
                             networks_in,
                             cur_hvProd_max: float = default_ctrld_hvProd_max,
                             ctrld_hvProd_max: float = default_ctrld_hvProd_max
                             ) -> dict:
    """ Create a Dict of variables that will be send to the parallel engines.

    Create a specific dictionary containing variable that will be send to the local
    space of the parallel engines.

    Parameters
    ----------
    df_pred_in : pandas.DataFrame
        Dataframe (Predicted values) of Total lower voltage producer, load demand and
        all the Higher voltage producer in lower level network.
    networks_in : oriClass.InitNetworks
        Networks initialized. An instance of :func:`oriClass.InitNetworks`, especially
        the output of the function :py:func:`oriFunctions.setNetwork_params`
    cur_hvProd_max : float, Optional, default = :data:`oriVariables.default_ctrld_hvProd_max`
        Current value of maximum output Power of the controlled HV producer (MW)
    ctrld_hvProd_max : float, Optional, default = :data:`oriVariables.default_ctrld_hvProd_max`
        Maximum fixed output of the Controlled Higher voltage producer (MW)

    Returns
    -------
    dict of dataframe
        The created dictionary with the its keys being
          `df_prodHT` :  pandas.DataFrame
              Dataframe containing the upscaled (based on ``cur_hvProd_max``) PV power
              of the Higher voltage  producers in lower level network.
          `df_prod_bt_total` :  pandas.DataFrame
              Dataframe of the upscaled (based on ``coef_add_bt``) total pv power of
              all lower voltage producer in the lower network.
          `df_cons_total` :  pandas.DataFrame
              Dataframe of the total load demand (consumption) in the lower level
              network
          `lowerNet_sgenDf_copy` :  pandas.DataFrame
              Dataframe of all the static generator in the lower network.

    Notes
    -----
    Add a checking function to make sure that ``cur_hvProd_max`` <= ``ctrld_hvProd_max``.
    If the constraint does not hold raise an exception. This makes sence because the
    variation of the controlled Hv Prod can't be higher than the maximum fixed output
    of the said Hv prod. As example, if ``ctrld_hvProd_max = 4 MW`, ``ctrld_hvProd_max``
    must be <= 4MW

    """

    # Instantiate parameters
    upNet_sum_max_lvProd = networks_in.get_upperNet_sum_max_lvProdLoad()[0]
    params_coef_add_bt = networks_in.get_params_coef_add_bt()
    ctrld_hvProd_name = networks_in.get_ctrld_hvProdName()

    # Check if coef_add_bt_dist is authorised
    checker.check_coef_add_bt_dist(params_coef_add_bt[1])

    # Check whether the input dataframe columns are in the expected order
    checker.check_networkDataDf_columnsOrder(df_pred_in)

    df_pred = df_pred_in.copy(deep=True)  # Create a copy of the input dataframe

    # If the last 2 digits of an elm of df_pred.columns is decimal, therefore the columns is
    # that of a HV producer
    hvProd_columns = [elm for elm in df_pred.columns if elm[-4:].isdecimal()]
    df_prodHT = df_pred[hvProd_columns]

    # Upscale HV production and the LV µ% production
    df_prodHT[[ctrld_hvProd_name]], df_prod_bt_total = _upscale_HvLv_prod(df_prodHT[[ctrld_hvProd_name]],
                                                                          df_pred[['Prod_BT']],
                                                                          ctrld_hvProd_max, upNet_sum_max_lvProd,
                                                                          cur_hvProd_max, params_coef_add_bt)

    # Define consumption df
    # TODO : Code a function to check the order of the input dataframe
    df_cons_total = df_pred.iloc[:, [0]]

    # Define a dict
    dict_df_sgenLoad = dict({'df_prodHT': df_prodHT,
                             'df_prod_bt_total': df_prod_bt_total,
                             'df_cons_total': df_cons_total,
                             'lowerNet_sgenDf_copy': networks_in.get_lowerNet_sgenDf_copy()})

    return dict_df_sgenLoad


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

def robustness(df_out_block_pfOpf,
               df_hvProd_noControl,
               cur_hvProd_max: float = default_ctrld_hvProd_max,
               ctrld_hvProd_max: float = default_ctrld_hvProd_max,
               auth_max_VriseHvBus: float = defAuth_hvBus_vRiseMax,
               combRnn_param: tuple = (None, None)
               ):
    """ Implements Robust control

    This is done by letting the controlled Hv Producer inject all its production in
    the network when no exceeding of :data:`oriVariables.defAuth_hvBus_vRiseMax` is
    predicted. Replacement is done in place i.e. in  ``df_out_block_pfOpf``.

    Parameters
    ----------
    df_out_block_pfOpf : pandas.DataFrame
        Predicted values. Output of the block pf/opf when the function is applied to
        a unique prediction model. First output of :py:func:`combineRnnPred` when the
        function is applied to a combination of RNN. In this case ``combRnn_param``
        **MUST** be included.
    df_hvProd_noControl : Dataframe
        Dataframe of controlled HV Prod  with no control.
    cur_hvProd_max : float, Optional
        Current maximum output Power of the HV producer (MW)
    ctrld_hvProd_max : float, Optional
        Maximum fixed output of the Controlled Higher voltage producer (MW)
    auth_max_VriseHvBus : float, Optional, Default = :py:data:`oriVariables.defAuth_hvBus_vRiseMax`
        Threshold of maximum voltage allowed on the HV buses of ``network``.
    combRnn_param: tuple, Optional, Default =(None, None)
        A tuple of parameter to use when applying the function to a combination of
        RNN.
            ``combRnn_param[0]`` : pandas.Dataframe,
                Dataframe of binary threshold i.e. the second output of
                :py:func:`combineRnnPred`.
            ``combRnn_param[1]`` : int or str
                Number of models which must agree on voltage rise above threshold
                before a command is sent to the controlled HV producer. This is
                the fourth input of :py:func:`combineRnnPred`.

    Raises
    ------
    Exception
        When ``df_out_block_pfOpf`` and ``combRnn_param`` do not concorde.


    """

    # Basically, we replace the value of the controlled HvProd by its own
    # value with No control when no voltage rise above the defined threshold is detected.

    # Check input concordance
    checker.check_robustnessParams(df_out_block_pfOpf, combRnn_param)

    # create new period index mask spanning from 07Am to 6PM
    per_index2 = (df_out_block_pfOpf.index
                  .to_timestamp()
                  .to_series()
                  .between_time('07:10', '18:50')
                  .index.to_period('10T'))

    # Get controlled HV prod Name
    ctrld_hvProdName = df_hvProd_noControl.columns[0]

    # Create a new df for hvProd
    hvProd_robust_df = pd.DataFrame(index=per_index2, columns=['hvProd_robust'])

    # Get into the dataframe data of hvProdWhen there is no control
    hvProd_robust_df.loc[per_index2, ['hvProd_robust']] = (df_hvProd_noControl.loc[per_index2, ctrld_hvProdName]
                                                           * cur_hvProd_max / ctrld_hvProd_max)

    if (combRnn_param[0] is None) and (combRnn_param[1] is None):  # If combRnn_param is not defined
        # Get a mask for the periods where a voltage rise above the threshold is predicted
        mask_vrise_per = df_out_block_pfOpf.loc[per_index2, 'max_vm_pu_pf'] > auth_max_VriseHvBus

    else:  # If combRnn_param is defined the robustness is being applied to a combination of RNN

        # Define variables' name as in :py:func:`combineRnnPred`.See the said function for more about
        # the parameters
        bin_thresh_df = combRnn_param[0]  # Binary threshold
        n_models = combRnn_param[1]  # Type of model to use

        if type(n_models) is str:  # If n_models  is a string
            # Create the mask based on the model's name
            mask_vrise_per = bin_thresh_df.loc[per_index2, n_models] == 1

        if type(n_models) is int:  # If n_model is int
            # Create the mask based on number of models that must agree
            mask_vrise_per = bin_thresh_df.Model_All.loc[per_index2] >= n_models

    # Replace the values of periods given by the mask by the value of hvProd with No control
    hvProd_robust_df[mask_vrise_per] = df_out_block_pfOpf.loc[per_index2[mask_vrise_per], [ctrld_hvProdName]]

    # Replace the values of hvProd_robust_df in the input df_out_block_pfOpf
    df_out_block_pfOpf.loc[per_index2, [ctrld_hvProdName]] = hvProd_robust_df.loc[per_index2, 'hvProd_robust']


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________

def block_prod(df_yTilde_opt,
               df_hvProd_noControl,
               cur_hvProd_max: float = default_ctrld_hvProd_max,
               ctrld_hvProd_max: float = default_ctrld_hvProd_max,
               starting_index: int = 0
               ):
    """ Implements block Prod.

    Implement block prod by making sure that the controlled HV producer can't
    inject into the lower network more than its actual production. Modify in
    place the input dataframe ``df_yTilde_opt``.

    Parameters
    ----------
    df_yTilde_opt : pandas.DataFrame
        :math:`\\tilde{Y}_{opt}(k)` after function :py:func:`robustness` has been
        applied to it.
    df_hvProd_noControl : Dataframe
        Dataframe of controlled HV Prod  with no control.
    cur_hvProd_max : float, Optional
        Current maximum output Power of the HV producer (MW)
    ctrld_hvProd_max : float, Optional
        Maximum fixed output of the Controlled Higher voltage producer (MW)
    starting_index : int, optional, default = 0
        Starting index. Important to use this starting index and set it to the
        length of a day in the case of the RNN. This is due to the fact that
        the prediction needs a whole day of data to be produced. Especially the
        first prediction must be that of the first index of the second day
        of the testing set since the whole first day (of the testing set) data
        is used.

    """
    # Get controlled HV prod Name
    ctrld_hvProdName = df_hvProd_noControl.columns[0]

    # create period index
    per_index2 = (df_yTilde_opt.index.to_timestamp()
                  .to_series()
                  .between_time('07:10', '18:50')
                  .index.to_period('10T'))

    # Upscale the HV prod when no control is applied
    df_hvProd_noControl_upscaled = (df_hvProd_noControl.loc[per_index2[starting_index:],
                                                            ctrld_hvProdName]
                                    * cur_hvProd_max / ctrld_hvProd_max)

    # Extract the part of the
    df_hvProd_controlled = df_yTilde_opt.loc[per_index2[starting_index:],
                                             ctrld_hvProdName]

    # Implement the actual prod
    df_yTilde_opt.loc[per_index2[starting_index:],
                      [ctrld_hvProdName]] = (np.minimum(df_hvProd_noControl_upscaled,
                                                        df_hvProd_controlled)
                                             )


# ____________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________


def setNetwork_params(upperNet_file: str,
                      lowerNet_file: str,
                      ctrld_hvProdName: str,
                      params_coef_add_bt: tuple = (None, None),
                      params_vRise: tuple = ((defAuth_hvBus_vRiseMax, defAuth_hvBus_vRiseMin),
                                             (defAuth_lvBus_vRiseMax, defAuth_lvBus_vRiseMin))
                      ) -> oriClass.InitNetworks:
    """

    Load both the lower (network used for optimization) and upper network, after which a
    configuration of the main  parameters to use for the simulations are done.

    Parameters
    ----------
    upperNet_file : str
        The upper Network file, with the appropriate extension (Must be present in the
        network_folder).\n
        Egg, 'ST LAURENT.p'.
    lowerNet_file : str
        The lower Network file, with the appropriate extension (Must be present in the
        network_folder)\n
        Egg, 'CIVAUX.p'.
    ctrld_hvProdName : str
        Name of the controlled HV producer in the Lower Network.\n
        Egg'P0100'.
    params_coef_add_bt : tuple
        coef_add_bt : float
            Value of the added output power for all the LV producers (MW) in the lower
            Network.
        coef_add_bt_dist : str
            How ``coef_add_bt`` is shared among the LV producers. Three choices are
            possible.
            None : default
                No upscaling is done
            "uppNet" :
                coef_add_bt is added to the Sum of maximum output of all lower voltage
                (LV) producers (MW) in the upper Network. In consequence, the LV
                producers on the lower network receive only a fraction of coef_add_bt.
            "lowNet" :
                coef_add_bt is added to the Sum of maximum output of all LV producers
                (MW) in the lower Network. In consequence, coef_add_bt is shared
                proportionally among all the LV producers on the lower network.
            "lowNet_rand" :
                coef_add_bt is shared proportionally among a randomly selected set of
                the LV producers on the lower Network. The randomly selected set
                consist of half of all LV producers on the lower Network.
    params_vRise : tuple
        ``params_vRise[0]`` : tuple
            vm_mu_max_hv : float
                Maximum authorised voltage rise of hv Buses on the Lower network
            vm_mu_min_hv : float
                Minimum authorised voltage rise of hv Buses on the Lower network
        ``params_vRise[1]`` : tuple
            Voltage Rise threshold associated with lower voltages buses
                vm_mu_max_lv : float
                    Maximum authorised voltage rise of lv Buses on the Lower network
                vm_mu_min_lv : float
                    Minimum authorised voltage rise of lv Buses on the Lower network

    Returns
    -------
    networks : :py:class:`oriClass.InitNetworks`
        An instance of the class :py:class:`oriClass.InitNetworks`

    Notes
    -----
    The option "lowNet_rand" of ``coef_add_bt_dist`` is not implemented yet.


    """

    # Extracts parameters
    coef_add_bt, coef_add_bt_dist = params_coef_add_bt
    vm_mu_max_hvBus, vm_mu_min_hvBus = params_vRise[0]
    vm_mu_max_lvBus, vm_mu_min_lvBus = params_vRise[1]

    # Check if coef_add_bt_dist is authorised
    checker.check_coef_add_bt_dist(coef_add_bt_dist)

    # Load lower and upper Network
    lowerNet = pp.from_pickle(f'{network_folder + lowerNet_file}')
    upperNet = pp.from_pickle(f'{network_folder + upperNet_file}')

    networks = oriClass.InitNetworks(upperNet, lowerNet, coef_add_bt, coef_add_bt_dist)  # Initialize networks

    networks.init_controlled_hvProd(ctrld_hvProdName)  # Initialize the controlled HVProd in the lowerNetwork

    # Extract HV and LV buses in the Lower Network
    lowerNet_hv_bus_df = networks.get_lowerNet_hv_bus_df(hvBus_voltage=default_hv_voltage)
    lowerNet_lv_bus_df = networks.get_lowerNet_lv_bus_df(lvBus_voltage=default_lv_voltage)

    # uppNet_sum_max_lvProdLoad = networks.get_upperNet_sum_max_lvProdLoad()  # To use later in functions

    # Extract the actives HV buses in the lower Network
    lowerNet_hv_activBus = networks.get_lowerNet_hvActivatedBuses(lowerNet_hv_bus_df.index)
    lowerNet_lv_activBus = networks.get_lowerNet_lvActivatedBuses(lowerNet_lv_bus_df.index)

    # Set the voltage rise threshold (both on Lv and HV buses) on the lower Network
    networks.lowerNet_set_vrise_threshold(lowerNet_hv_activBus, vm_mu_min_hvBus, vm_mu_max_hvBus)
    if coef_add_bt_dist in ['lowNet', 'lowNet_rand']:  # Set Vrise on lv buses only if coef_add_bt_dist is in list
        networks.lowerNet_set_vrise_threshold(lowerNet_lv_activBus, vm_mu_min_lvBus, vm_mu_max_lvBus)

    # Add negative cost to usability of controlled Sgen so its usage can be maximized while
    # respecting the constraints on the network
    # HV PROD
    ctrld_hvProd_index = networks.get_ctrld_hvProdName(return_index=True)[1]
    cost_sgen_ctrlHvProd = pp.create_poly_cost(lowerNet, ctrld_hvProd_index, 'sgen', cp1_eur_per_mw=-1)
    # LV PROD
    ctrld_lvProd_index = networks.get_ctrld_lvProdName(return_index=True)[1]
    [pp.create_poly_cost(lowerNet, cur_ctrld_lvProd_ind, 'sgen', cp1_eur_per_mw=-2)
     for cur_ctrld_lvProd_ind in ctrld_lvProd_index]

    networks.create_lowerNet_sgenDf_copy()  # Create a copy of the Sgens in the lower networks

    return networks


def par_block_pfOpf(par_engines,
                    pred_model_f: str=None
                    ) -> pandas.DataFrame:
    """ Block PF/OPF using parallels engines.

    Execute a power flow, an optimal power flow, or both depending on the ``opf_status``
    extracted from the :py:class:`oriClass.CreateParEngines` object ``par_engines``. When
    ``opf_status`` is `` "Both" ``, the function is used as the block PF/OPF. when ``opf_status``
    is  ``False``, the function is used as the block PF.

    Parameters
    ----------
    par_engines:  ipyparallel.cluster
        Parallel engines object,, Instance of  :py:class:`oriClass.CreateParEngines`.
    pred_model_f: str, Optional, Default = None
        'Pers' ==> Using persistence model.

    Returns
    -------
    pandas.DataFrame
        Output of the function :py:func:`oriClass.CreateParEngines.get_results_asDf`.

    Warnings
    --------
    DO NOT CALL this function from the module :py:mod:`oriFunctions`.
    Given that the function makes use of the magic ``%%px`` implemented by ipyparallel,
    it has to be call through a conversion of a jupyter notebook into a module. See
    :ref:`examples` or `SensAnalysisP0100 <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/SensAnalysisP0100.ipynb>`_
    for usage.


    .. examples:

    Examples
    --------
    >>> from fnfrnb import NotebookLoader              # Import NotebookLoader from module
    ...                                                # fnfrnb i.e. Function From Notebook

    >>>  oriPar = NotebookLoader().load_module('parFn') # Create the module oriPar based on
    ...                                                # the content of `parFn.ipynb`

    Importing parFn.ipynb content as a module

    >>> oriPar.par_block_pfOpf(par_engines, pred_model)# Call the parallel block PF/OPF

    """

    # see '../Notebooks/parFn.ipynb' for the content of the function
    pass

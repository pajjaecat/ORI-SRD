# TODO: FInd a way to use the function already defined in OriFunctions in the parallel engines 


import numpy
import pandapower
import pandas
from tqdm import tqdm  # Profiling

pd = pandas
np = numpy
pp = pandapower

# Import variables that each parallel engine will use

Δt = 1 / 6  # Time frequency 10mn ==> 1Hour/6
# Default Authorised voltage rise on hv Bus on lower Network
defAuth_hvBus_vRiseMax = 1.025
defAuth_hvBus_vRiseMin = 0.95
defAuth_lvBus_vRiseMax = 1.075
defAuth_lvBus_vRiseMin = 0.95

# Default Authorised voltage on hv and Lv Bus on lower Network
default_hv_voltage = 20.6
default_lv_voltage = 0.4

default_ctrld_hvProd_max = 4.0


# ___________________________________________________________________________________________________________________
# ------------------------------------------------------------------------------------------------------------------
# ___________________________________________________________________________________________________________________

def run_powerflow(network: pandapower.auxiliary.pandapowerNet,
                  lowNet_hv_activBus: list,
                  sum_max_p_mw_upperNet: tuple,
                  dict_df_sgenLoad: dict,
                  opf_status=False):
    """
    Return a list of maximum voltage on the network for each period given by the index 
    of element in 

    Initialise the parameters of the network

    Parameters:
    ----------
    network: Pandapower network
        The network to beimulation consider ;
    dict_df_sgenLoad: dict 
        Dictionary containing data (as dataframe i.e indexed by each period of 
        the considered year for the first 3 imputs) of the 
        + df_prodHT            => Power of all Higher voltage producers in the lower network 
        + df_prod_bt_total     => Total power of all lower voltage producers seen from the higher Network
        + df_cons_total        => Total Load demand seen from the higher Network
        + lowerNet_sgenDf_copy => Copy of all the static generator in the lower network
    sum_max_p_mw_upperNet: tuple
        Sum of maximum power seen from the bigger network (here, saint laurent 
        compared to the subnetwork civaux)
        + Of all BT energy producers => sum_max_input[0]
        + of all Load in the network => sum_max_input[1] 
    lowNet_hv_activBus: list
        list of all Hv bus activated in the concerned network
    ofp_status: Boolean, optional, default False
        Wether the maximum voltage rise on the lower network buses  is computed after a power 
        flow, an optimal power flow  or both
        + False  =>  **pandapower.runpp(net)**
        + True   =>  **pandapower.runopp(net)**, ofp_status = True
        + 'Both' =>  A power flow is run. Only when the result i.e. the voltage rise detected 
                     on hv Prod Buses max_vm_pu > auth_max_VriseHvBus, is the  optimal  power 
                     flow run.


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
                                                                             dict_df_sgenLoad, opf_status)
            list_max_vm_pu.append(max_vm_pu)
            list_sgen_HT.append(hvProd_afterOPF)

        else:  # Run simple power flow
            hvProd_pf, lvProd_pf = run_powerflow_at(network, cur_period,
                                                    lowNet_hv_activBus,
                                                    sum_max_p_mw_upperNet,
                                                    dict_df_sgenLoad, opf_status)

            list_max_vm_pu.append(hvProd_pf)
    # Return depends on ofp_status
    if opf_status:
        return list_max_vm_pu, list_sgen_HT
    else:
        return list_max_vm_pu


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
    check_var_concordance(opf_status, pred_model)

    # -- GT1
    if pred_model == 'Pers':  # if  the prediction model is the persistance,
        cur_period = cur_period - 1

    # Initialize the network. See the corresponding function for more explanation
    initLowerNet_at(network, cur_period, sum_max_p_mw_upperNet, dict_df_sgenLoad)

    # Get the maximum voltage magnitude of all activated bus to a list. See the
    #                               corresponding function for more explanation
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
        #  reset cur_period to its initial value using <cur_period+1> before ruturning the results
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
        # If the maximum voltage on buses is above the authorized threshold, run optimal power flow
        if cur_max_VriseHvBus > auth_max_VriseHvBus:
            cur_max_VriseHvBus = max_vm_pu_at(network, cur_period, lowNet_hv_activBus,
                                              dict_df_sgenLoad, True)

        # Get the value of HV and LV producers after optimal flow.
        hvProd_afterOPF = list(network.res_sgen[network.sgen.name.notna()].p_mw)
        lvProd_afterOPF = list(network.res_sgen[network.sgen.name.isna()].p_mw)

        # Depending on the prediction model parameter the return is different ----------------
        # For <pred_model = 'Pers'> given that at GT1 the <cur_period = cur_period-1> one must
        #  reset cur_period to its initial value using <cur_period+1> before ruturning the results
        if pred_model == 'Pers':
            return [max_vm_pu_pf, cur_max_VriseHvBus], (hvProd_afterOPF, lvProd_afterOPF), cur_period + 1
        else:
            return [max_vm_pu_pf, cur_max_VriseHvBus], (hvProd_afterOPF, lvProd_afterOPF), cur_period

    elif opf_status is False:  # Run normal power flow  ***************************************************
        return max_vm_pu_at(network, cur_period, lowNet_hv_activBus, dict_df_sgenLoad, opf_status), cur_period

    else:
        raise ValueError('<opf_status> must be either of [True, False, "Both"]')


# __________________________________________________________________________________________________________________
# -----------------------------------------------------------------------------------------------------------------
# __________________________________________________________________________________________________________________

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


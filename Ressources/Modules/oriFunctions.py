""" 
OriFunctions - Python library with List of all function used in the `tutorials <https://github.com/pajjaecat/ORI-SRD/tree/main/Ressources/Notebooks>`_


.. code-block:: python


    # Import oriFunctions
    import oriFunctions

"""
 
import pandapower, pandas, numpy, ipyparallel
from tqdm import tqdm  # Profiling

# Import all variables from module oriVariables
from oriVariables import (network_folder, excel_folder, attr_list,
                          defAuth_hvBus_vRiseMax, defAuth_hvBus_vRiseMin,
                          defAuth_lvBus_vRiseMax, defAuth_lvBus_vRiseMin,
                          default_ctrld_hvProd_max, 
                          default_hv_voltage, default_lv_voltage)

pd = pandas
np = numpy 
pp = pandapower
ipp = ipyparallel



##############################         FUNCTIONS          #########################################

def readAndReshape_excelFile(f_name:str, 
                             folder_name=excel_folder, 
                             n_row2read:int=None):
    """
    Read and reshape in a one dimension array (that is returned) the excel file given by f_name


    Parameters 
    ---------- 
    f_name: str
        Name of the file to load (with the correct extension)
    folder_name: str
        Location of the folder where the file is present
    n_row2read : Int (default=0) 
         Numbers of rows to read in the excel file.

    """

    filename = f"{folder_name}{f_name}"
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
# -----------------------------------------------------------------------------------------------------------
# ____________________________________________________________________________________________________________

def check_bus_connection(network, bus_number, attr_list_in:dict=attr_list):
    """
    Check and print the connection between a bus number and all the elements in the lower network.

    Parameters
    ----------
    network: pandapower network
        The network that has to be investigated
    bus_number: list of int
        The number of the concerned bus(ses)
    attr_list_in: list of String tuple
        Each tuple in the list represent the attribute of the attribute to look for
        Ex: attr_list_in[0] = ('bus', 'name') ==> network.bus.name must be accessed
    
    """

    for cur_bus in bus_number:  # For each bus
        for attribute in attr_list_in:  # For each tuple in the attibute list
            netsub = getattr(network, attribute[0])
            netsub_sub = getattr(netsub, attribute[1])

            if len(netsub[netsub_sub == cur_bus]) != 0:  # If there is some elements
                print(
                    f'----------******   Bus {cur_bus} net.{attribute[0]}.{attribute[1]} ******-------')
                print(netsub[netsub_sub == cur_bus], '\n')
        print('\n')


# ____________________________________________________________________________________________________________
# -------------------------------------------------------------------------------------------------------------
# _____________________________________________________________________________________________________________

def run_powerflow(network: pandapower.auxiliary.pandapowerNet,
                  lowNet_hv_activBus: list,
                  sum_max_p_mw_upperNet: tuple,
                  dict_df_sgenLoad: dict,
                  opf_status=False):
    """
    Return a list of maximum voltage on the network for each period given by the index 
    of element in 

    Initialise the parameters of the network

    Parameters
    ----------
    network: Pandapower network
        The network to beimulation consider ;
    dict_df_sgenLoad: dict 
        Dictionary containing data (as dataframe i.e indexed by each period of 
        the considered year for the first 3 imputs) of the 
        + df_prodHT            => Power of all Higher voltage producers in the lower network 
        + df_prod_bt_total     => Total power of all lower voltage producers seen  from  the 
                                  upper Network
        + df_cons_total        => Total Load demand seen from the upper Network
        + lowerNet_sgenDf_copy => Copy of all the static generator (hv & lV) in the lower net
    sum_max_p_mw_upperNet: tuple
        Sum of maximum power seen from the upper network (here, saint laurent 
        compared to the lower network civaux)
        + Of all Lower voltage producers energy producers => sum_max_input[0]
        + of all Load in the network                      => sum_max_input[1] 
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
    list_sgen_HT = []    # Actual HT generators power after optimal flow

    # Initiate parameters from inputs
    df_prodHT = dict_df_sgenLoad['df_prodHT']

    # Initialise the network and get the maximum value for each period
    for cur_period in tqdm(df_prodHT.index):
        print(cur_period)
        if opf_status:  # Run optimal power flow
            print('True or Both')
            
            max_vm_pu, (hvProd_afterOPF, lvProd_afterOPF) = run_powerflow_at(network, cur_period,
                                                                            lowNet_hv_activBus,
                                                                            sum_max_p_mw_upperNet,
                                                                            dict_df_sgenLoad, 
                                                                             opf_status)
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





# ____________________________________________________________________________________________________________
# -----------------------------------------------------------------------------------------------------------
# ____________________________________________________________________________________________________________

def run_powerflow_at(network:pandapower.auxiliary.pandapowerNet,
                     cur_period: pandas._libs.tslibs.period,
                     lowNet_hv_activBus: list,
                     sum_max_p_mw_upperNet: tuple,
                     dict_df_sgenLoad: dict,
                     auth_max_VriseHvBus: float=defAuth_hvBus_vRiseMax,
                     opf_status: (bool or str) =False, 
                     pred_model: str=None):
    """
    Run Power flow or optimal power flow depending on 'opf_status'

    Parameters
    ----------
    network: Pandapower network
        The network to consider ;
    cur_period: Panda period
        The current period to investigate
    lowNet_hv_activBus: list
        list of all Hv bus activated in the concerned network
    sum_max_p_mw_upperNet: tuple
        Sum of maximum power seen from the upper network (here, saint laurent 
        compared to the lower network civaux)
        + Of all Lower voltage producers energy producers => sum_max_input[0]
        + of all Load in the network                      => sum_max_input[1] 
    dict_df_sgenLoad: dict 
        Dictionary containing data (as dataframe i.e indexed by each period of 
        the considered year for the first 3 imputs) of the 
        + df_prodHT            => Power of all Higher voltage producers in the lower network 
        + df_prod_bt_total     => Total power of all lower voltage producers seen  from  the 
                                  upper Network
        + df_cons_total        => Total Load demand seen from the upper Network
        + lowerNet_sgenDf_copy => Copy of all the static generator (hv & lV) in the lower net
    auth_max_VriseHvBus: float, optional
        Threshold of maximum voltage allowed on  the network. Is used only when `ofp_status` 
        is 'Both';
    ofp_status: Boolean, optional, default False
        Wether the maximum voltage rise on the lower network buses  is computed after a power 
        flow, an optimal power flow  or both
        + False  =>  **pandapower.runpp(net)**
        + True   =>  **pandapower.runopp(net)**, ofp_status = True
        + 'Both' =>  A power flow is run. Only when the result i.e. the voltage rise detected 
                     on hv Prod Buses max_vm_pu > auth_max_VriseHvBus, is the  optimal  power 
                     flow run.
    pred_model: str, optional
        Which kind of prediction model to use for the all the variables to predict at current 
        period.
        + None: No prediction model is used,
        + Pers  =>  Persistence model i.e. val(k)= val(k-1)
        
    Returns 
    -------
    Depends on 'ofp_status' 
    False ==>                cur_max_VriseHvBus
    True  ==>                cur_max_VriseHvBus, (hvProd_afterOPF, lvProd_afterOPF), cur_period
    'Both'==> [max_vm_pu_pf, cur_max_VriseHvBus],(hvProd_afterOPF, lvProd_afterOPF), cur_period
    where
        cur_max_VriseHvBus: float 
            Maximum voltage rise detected on all the Hv buses on the lower Network
        hvProd_afterOPF: list 
            List (in the order  in which they appear in the pandapower network sgen table) of 
            the optimal power that each hv producer on the lower net must inject in order  to 
            satisfy  the auth_max_VriseHvBus.
        lvProd_afterOPF: list
            List (in the order  in which they appear in the pandapower network sgen table) of 
            the optimal power that each lv producer on the lower net must inject in order  to 
            satisfy the auth_max_VriseHvBus.
        cur_period: panda peridod
            the period at which the pf/OPF is run
             
    Notes
    -----
    For the moment auth_max_VriselvBus i.e. the authorised voltage rise on the lv
    buses constraint is  considered only when the Voltage rise on the hv buses  is greater 
    than auth_max_VriseHvBus. Simply put, as long as no voltage rise above auth_max_VriseHvBus
    is detected one does not care about the value of the voltage rise on the lv buses.
    TODO :Considered the auth_max_VriselvBus to run an opf.   
    
    """

  
    # -- GT1
    if pred_model == 'Pers': # if the the prediction model is the persistance,
        cur_period = cur_period-1
        
        
    # Initialize the network. See the corresponding function for more explanation
    initLowerNet_at(network, cur_period, sum_max_p_mw_upperNet, dict_df_sgenLoad)

    # Get the maximum voltage magnitude of all activated bus to a list. See the 
    #                               corresponding function for more explanation
    if opf_status == True:  # Run optimal power flow ******************************************

        print('True or Both')

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
            return cur_max_VriseHvBus, (hvProd_afterOPF, lvProd_afterOPF), cur_period+1
        else:
            return cur_max_VriseHvBus, (hvProd_afterOPF, lvProd_afterOPF), cur_period

    elif opf_status == 'Both':# Run normal and depending on the situation, also optimal power flow  **
        # run power flow first 
        cur_max_VriseHvBus = max_vm_pu_at(network, cur_period, lowNet_hv_activBus,
                                          dict_df_sgenLoad, False)
        max_vm_pu_pf = cur_max_VriseHvBus # Save the maximum voltage given by the power flow 
                                          # before optimizing
        # If the maximum voltage on buses is above the authorized threshold, run optimal power flow
        
        if cur_max_VriseHvBus > auth_max_VriseHvBus:
            print(cur_period,'True in Both')
            
            cur_max_VriseHvBus = max_vm_pu_at(network, cur_period, lowNet_hv_activBus, 
                                                                      dict_df_sgenLoad, True)

        # Get the value of HV and LV producers after optimal flow. 
        hvProd_afterOPF = list(network.res_sgen[network.sgen.name.notna()].p_mw)
        lvProd_afterOPF = list(network.res_sgen[network.sgen.name.isna()].p_mw)


        # Depending on the prediction model parameter the return is different ----------------
        # For <pred_model = 'Pers'> given that at GT1 the <cur_period = cur_period-1> one must
        #  reset cur_period to its initial value using <cur_period+1> before ruturning the results
        if pred_model == 'Pers': 
            return [max_vm_pu_pf, cur_max_VriseHvBus], (hvProd_afterOPF, lvProd_afterOPF), cur_period+1
        else:
            return [max_vm_pu_pf, cur_max_VriseHvBus], (hvProd_afterOPF, lvProd_afterOPF), cur_period

    elif opf_status == False :  # Run normal power flow  ***************************************************
        return max_vm_pu_at(network, cur_period, lowNet_hv_activBus, dict_df_sgenLoad, opf_status)
    
    else : 
        raise ValueError('<opf_status> must be either of [True, False, ''Both'']' )      





# ____________________________________________________________________________________________________________
# -----------------------------------------------------------------------------------------------------------
# ____________________________________________________________________________________________________________
        


def initLowerNet_at(network: pandapower.auxiliary.pandapowerNet,
                    cur_period: pandas._libs.tslibs.period,
                    sum_max_p_mw_upperNet: tuple,
                    dict_df_sgenLoad: dict):
    """

    Initialise the parameters of the network at the current period

    Parameters
    ----------
    network: Pandapower network
        The lower level network concerned ;
    cur_period: Pandas period
        The current period to investigate;
    sum_max_p_mw_upperNet: tuple
        Sum of maximum power seen from the upper level network (here, saint laurent 
        compared to the lower level network civaux)
        + Of all BT energy producers => sum_max_input[0]
        + of all Load in the network => sum_max_input[1] 
    dict_df_sgenLoad: dict 
        Dictionary containing data (as dataframe i.e indexed by each period of 
        the considered year for the first 3 imputs) of the 
        + df_prodHT            => Power of all Higher voltage producers in the lower network 
        + df_prod_bt_total     => Total power of all lower voltage producers seen  from  the 
                                  upper Network
        + df_cons_total        => Total Load demand seen from the upper Network
        + lowerNet_sgenDf_copy => Copy of all the static generator (hv & lV) in the lower net


    """
    ##  TODO : Give only the data of the current period to
    ##  the function instead of that of the whole year

    # Initiate parameters to be used within funtion
    upNet_sum_max_lvProd = sum_max_p_mw_upperNet[0]
    upNet_sum_max_load = sum_max_p_mw_upperNet[1]

    df_prodHT = dict_df_sgenLoad['df_prodHT']
    df_prod_bt_total = dict_df_sgenLoad['df_prod_bt_total']
    df_cons_total = dict_df_sgenLoad['df_cons_total']
    df_lowNetSgen_cp = dict_df_sgenLoad['lowerNet_sgenDf_copy']
    
    # Initalise HT producers 
    network.sgen.p_mw[network.sgen.name.notna()] = df_prodHT.loc[cur_period].values

    # Create a mask of all controlled LV producer in the lower Network
    mask_ctrld_lvProd = df_lowNetSgen_cp.name.isna() & df_lowNetSgen_cp.controllable
    
    # Initialized maximum power of controllable LV producers
    network.sgen.loc[mask_ctrld_lvProd, 'max_p_mw'] = (df_lowNetSgen_cp[mask_ctrld_lvProd].
                                                                                   max_p_mw)
    # Initialize Bt producers
    prod_bt_total_1mw = df_prod_bt_total.loc[cur_period].values/upNet_sum_max_lvProd
    network.sgen.p_mw[network.sgen.name.isna()] = (network.sgen.max_p_mw[network.sgen.name.isna()] 
                                                   *prod_bt_total_1mw
                                                  )
    # Initialize Loads
    load_total_1mw = df_cons_total.loc[cur_period].values/upNet_sum_max_load
    network.load.p_mw = (network.load.max_p_mw*load_total_1mw)

    # Work with julia Power model since the load is zero
    # network.load.p_mw = (network.load.max_p_mw*df_cons_total.loc[cur_period].
    # values*0/upNet_sum_max_load)


# _____________________________________________________________________________________________________________
# -------------------------------------------------------------------------------------------------------------
# _____________________________________________________________________________________________________________

def max_vm_pu_at(network: pandapower.auxiliary.pandapowerNet,
                 cur_period: pandas._libs.tslibs.period,
                 lowNet_hv_activBus: list,
                 dict_df_sgenLoad: dict,
                 opf_status: (bool or str) = False ):
    """

    Return the maximum voltage over all the higher voltages active buses in the network for the current 
    period.

    Parameters
    ----------
    network: Pandapower network
        The network
    cur_period: Panda period
        The current period to investigate;
    lowNet_hv_activBus: List
        List of all the higher voltage activated bus in the network
    dict_df_sgenLoad: dict 
        Dictionary containing data (as dataframe i.e indexed by each period of 
        the considered year for the first 3 imputs) of the 
        + df_prodHT            => Power of all Higher voltage producers in the lower network 
        + df_prod_bt_total     => Total power of all lower voltage producers seen  from  the 
                                  upper Network
        + df_cons_total        => Total Load demand seen from the upper Network
        + lowerNet_sgenDf_copy => Copy of all the static generator (hv & lV) in the lower net
    ofp_status: Boolean, optional, default False
        Wether the maximum voltage rise on the lower network buses  is computed after a power 
        flow, an optimal power flow  or both
        + False  =>  **pandapower.runpp(net)**
        + True   =>  **pandapower.runopp(net)**, ofp_status = True
        + 'Both' =>  A power flow is run. Only when the result i.e. the voltage rise detected 
                     on hv Prod Buses max_vm_pu > auth_max_VriseHvBus, is the  optimal  power 
                     flow run.
                     
--------------------------------------------------------------------------------------------------------
    Notes
    ------
    Return the maximum voltage rise over all the LV buses in the lower network for the current 
    instant. In this case one needs to add as input to the function the  net_lv_activBus 
    list as well. Hence one can replace the lowNet_hv_activBus by a tuple 
    of (lowNet_hv_activBus, uppNet_lv_activBus). 

    
    """

    # Initiate parameters from input
    df_prodHT = dict_df_sgenLoad['df_prodHT']
    df_lowNetSgen_cp = dict_df_sgenLoad['lowerNet_sgenDf_copy']
    
    # Create a mask of all controlled LV producer in the lower Network
    mask_ctrld_lvProd = df_lowNetSgen_cp.name.isna() & df_lowNetSgen_cp.controllable

    
    
    if opf_status:  # If status is True or both
        
        # Extract the name and the index of the controlled Higher voltage producer.
        # This supposed the there is a controllable column in the network, this controllable column 
        # is true for the controlled HV producer
        hvSgen_df = network.sgen[network.sgen.name.notna()]
        ctrld_hvProd_name = list(hvSgen_df[hvSgen_df.controllable].name)[0]
        ctrld_hvProd_ind = list(hvSgen_df[hvSgen_df.controllable].index)[0]
        
        # update 
        # For optimal flow, given that the sgen P0100 is contollable the optimization 
        # result is to draw the maximum power  with no regard to the actual power provided 
        # at each instant. To eliavate this problem we would rather initialize the maximum 
        # power of the said  producer with the actual prooduction. 
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
# -----------------------------------------------------------------------------------------------------------
# ____________________________________________________________________________________________________________            
            
def improve_persinstence(per_extracted_res_df: pandas.core.frame.DataFrame, 
                         prodHT_df: pandas.core.frame.DataFrame,
                         auth_vm_mu_max: float, 
                         h_start_end = ['11:00','14:00']):
# Implement : * Inject all the production as long as max_vm_pu_pf < vm_mu_max, i.e. 
# no voltage rise is detected 
    """
    Improve the results given by the persistence model. If a voltage rise is not predicted by 
    the persistence model at a certain period, the controllable sgens is allowed to inject all 
    its power into the grid. Otherwise the energy producer can inject at most the predicted power 
    by the persistence model. 


    Parameters
    ----------
    per_extracted_res_df: dataframe
        Result given by the persistence model. 
        Output of <<myFunction.extract_par_results(par_results_pers, *args).
    df_prodHT: Dataframe
        Dataframe containing data of all the HT producers in the network
    auth_vm_mu_max: 
        Threshold of maximum voltage allowed on the network. 


    Returns
    -------
    per_improved_res: dataframe
    Output the improve persistence model improve by the previoulsy described strategy
    
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
    working_df = per_improved_res.between_time(h_start_end[1],h_start_end[0])
    
    # Extract index of instances where no voltage rise is detected ignoring the last one 
    # because not present in the inital df df_prodHT
    var_index = working_df[working_df.max_vm_pu_pf<=auth_vm_mu_max].index.to_period('10T')[:-1]
    
    # remplace the prediction from the persistence model with the actual production since
    # no voltage rise is detected at these periods
    per_improved_res_out.P0100[var_index] = prodHT_df.P0100[var_index]
    

    return per_improved_res_out, var_index
    

    
 

    
# ____________________________________________________________________________________________________________
# -----------------------------------------------------------------------------------------------------------
# ____________________________________________________________________________________________________________          
def prediction_bloc(rnn_model, fitting_scaler, history, scaler_features=None ):
    """
    Prediction bloc: Predict the values () of the next period based on the RNN (LSTM). 


    Prameters 
    ----------
    rnn_model: Recurent neural network; 
        The model that will be used to predict the value at the next period. 
    fitting_scaler: Scaler
        Scaler parameters that are used to transform the training data set 
        fed to the RNN. 
    history: Non scaled history of the Electrical network: 
    scaler_features : Scaler to use for prediction when the number of 
        variables to predict is different from the number of features



    Returns
    ------
    List
     (1): Prediction of the interest variable 
     (2): Period for wich the prediction is done

    
    """
    
    history_last_ind = history.index[-1]              # Get index of the last period of history
    in_shape = tuple([1]) + rnn_model.input_shape[1:] # define input shape for the RNN
    
    # Scaled the input  based on the fitting scaler 
    scaled_history = fitting_scaler.transform(history).reshape(in_shape)  
    
    pred = rnn_model.predict(scaled_history, verbose=False)  # prediction
    
    # inverse transform the prediction
    if scaler_features is None:   pred_inv_trans = fitting_scaler.inverse_transform(pred)  
    else : pred_inv_trans = scaler_features.inverse_transform(pred) 
    
    # Return the prediction of the RNN and the time period associated ()
    return pred_inv_trans, history_last_ind+1






# ____________________________________________________________________________________________________________
# -----------------------------------------------------------------------------------------------------------
# ____________________________________________________________________________________________________________          
def predictionBin_bloc(rnn_model, fitting_scaler, history, sig_thresh=0.5):
    """
    Prediction bloc: Predict the values () of the next period based on the RNN (LSTM). 


    Prameters 
    ----------
    rnn_model: Recurent neural network; 
        The model that will be used to predict the value at the next period. 
    fitting_scaler: Scaler
        Scaler parameters that are used to transform the training data set 
        fed to the RNN. 
    history: 
        Non scaled history of the Electrical network:
    sig_thresh: 
        Sigmoid threshold



    Returns
    -------
    List 
     (1): Prediction of the interest variable 
     (2): Period for wich the prediction is done

    
    """
    
    history_last_ind = history.index[-1]              # Get index of the last period of history
    in_shape = tuple([1]) + rnn_model.input_shape[1:] # define input shape for the RNN
    
    # Scaled the input  based on the fitting scaler 
    scaled_history = fitting_scaler.transform(history).reshape(in_shape)  
    
    pred = rnn_model.predict(scaled_history, verbose=False)  # prediction
    
    
    pred_bin = (pred>sig_thresh).astype(int) # convert prediction into a binary variablethe prediction
    
    # Return the prediction of the RNN and the time period associated ()
    return pred_bin[0][0], history_last_ind+1






# ____________________________________________________________________________________________________________
# -----------------------------------------------------------------------------------------------------------
# ____________________________________________________________________________________________________________          
def robustPred(model_Vrise_dict, hvProd_noControl, P0100_opt_model1, 
               auth_vm_mu_max:float = defAuth_hvBus_vRiseMax, 
               n_models = None ):
    """
    Define Robust prediction bloc: 


    Prameters
    ----------
    model_Vrise_dict: Dict
        Dictionary of the voltage rise for each model 
    hvProd_noControl : pandas dataframe
        Values of the controlled Generator P0100 when no controled is applied
    P0100_opt_model1 : pandas Dataframe. Partial output of function <<extract_par_results>>
        Optimal value of P0100 at the end of bloc PF/OPF of model1. This is the 
        command value to send to the said producer when the robustPred 
        predicts a voltage rise above the threshold vm_mu_max.
    auth_vm_mu_max: Threshold of maximum voltage on the network
    n_models: Int or string
        Int: Number of models which must agree on voltage rise above threshold before
        a command is set to P0100
        ** 1: At Least one of the models
        ** 2: At least two of the models
        ** 3: All three models
        String: 
            Name of the Model which voltage rise above threshold prediction is considered
                'Modelx' where x in {1,2,3}


    Returns 
    -------
    new_p0100_df: panda dataframe
        y_optimal after combined model

    """
    

    # Extract model voltage rise from Input dictionary 
    model1_Vrise, model2_Vrise, model3_Vrise = (model_Vrise_dict['Model1'], 
                                                model_Vrise_dict['Model2'], 
                                                model_Vrise_dict['Model3'])
    
    mask_per2work = model1_Vrise.index # Get index of the considered period
    vect_int = np.vectorize(int)       # vectorized version of int

    # Create an empty dataframe i.e. binary threshold 
    bin_thresh_df = pd.DataFrame(index=mask_per2work)

    # add the binary output of three models to the created df
    bin_thresh_df[['Model3']] = model3_Vrise.values
    bin_thresh_df[['Model2']] = vect_int(model2_Vrise>auth_vm_mu_max)
    bin_thresh_df[['Model1']] = vect_int(model1_Vrise>auth_vm_mu_max)

    # Combined_output of all models
    bin_thresh_df[['Model_All']] = np.array(bin_thresh_df.sum(axis=1)).reshape((-1,1))

    
    # Create a new dataframe for the controlled SGEN based on its real values 
    new_p0100_df = hvProd_noControl.loc[mask_per2work, ['P0100']]

    
    if type(n_models) is str :# If n_model is a string
        if n_models in model_Vrise_dict.keys(): # Check if the sting input is in the model Dict
            vrise_true_mask = bin_thresh_df[n_models] == 1 # Create the mask using only the 
                       # period where the concerned model predict there is an voltage rise
        else: raise ValueError('Since <n_models> is a string it must be be either of', 
                               list(model_Vrise_dict.keys()))      
            
    elif type(n_models) is int: # If n_model is int 
        if n_models <= 3:
            # Create mask of instants where at least n models agrees on voltage rise above threshold 
            vrise_true_mask = bin_thresh_df.Model_All>= n_models 
        else: raise ValueError('Since <n_models> is an int it must be defined such that 0 < n_models <= 3 ')
    
    else: raise ValueError('<n_models> is the wrong type. Must either be an int or a string')

    # Use vrise_true_mask to insert predicted values given by model1 at the concerned instants 
    new_p0100_df[vrise_true_mask] = P0100_opt_model1.loc[mask_per2work].loc[vrise_true_mask, ['P0100']]
    
    return new_p0100_df, bin_thresh_df









def _upscale_HvLv_prod(prod_hv2upscale_df, prod_lv2upscale_df, 
                      ctrld_hvProd_max, upNet_sum_max_lvProd,
                      cur_hvProd_max:int=0, 
                      params_coef_add_bt: tuple=(None,None)
                      ):
    """
    Upscale  both the controled Higher voltage(HV) producer (P0100) in the lower network (civeaux) 
    and the total Lower voltage (LV) production. Check the parameter 'coef_add_bt_dist' to choose how 
    the upscaling is done on the LV production.
    THis mean the BT producer on the lower network receives only a fraction of the added BT production. 
    See function upscale_HvLv_prod() for the version of the function where the BT producer receive
    all the Added BT prod


    Parameters 
    -----------
    prod_hv2upscale_df: pd.dataframe
        dataframe of the HV prod to upscale i.e. P0100
    prod_lv2upscale_df: pd.dataframe 
        dataframe of the total LV producers output (That must be increased) i.e. Prod_BT
    ctrld_hvProd_max: float
        Maximum fixed output of the Controlled Higher voltage producer (MW)
    upNet_sum_max_lvProd: float
        Sum of maximum output of all lower voltage (LV) producers (MW) in the upper Network
        TODO: Get upNet_sum_max_lvProd from oriClass.InitNetwork() instance 
    cur_hvProd_max: Int (default=0) 
        Current value of maximum output Power of the HV producer (MW)
    params_coef_add_bt:tuple
        Parameters associated with how the upscaling of the total LV production is done. See doc
        # oriClass.InitNetworks(*args) for more information
        (1): coef_add_bt 
        (2): coef_add_bt_dist
        
    Returns
    -------
        The upscaled version of the HV producer and LV

    """

        
    ###  ----------- Upscale P0100 the controlable HT producer  ---------------  ###
    prodHT_P0100_1mw_df = prod_hv2upscale_df/ctrld_hvProd_max      # Rescale the total HT production for 1mw
    upscaled_prodHT_P0100_df = cur_hvProd_max*prodHT_P0100_1mw_df # Redefine the HT production based on the rescale


    # Get the parameters coef_add_bt and coef_add_bt_dist 
    coef_add_bt = params_coef_add_bt[0]
    coef_add_bt_dist = params_coef_add_bt[1]
    
    
    ###  ----------- upscale BT production ---------------  ###
    # Only the upscaling on the upper Network is done here. For the other cases, 
    # i.e. 'lowNet' and 'lowNet_rand', see oriClass.InitNetworks(*args)
    upscaled_prod_bt_total_df = prod_lv2upscale_df
    if coef_add_bt_dist == 'uppNet' :
    
        # Rescale the total BT production for 1mw
        prod_bt_total_1mw_df = prod_lv2upscale_df/upNet_sum_max_lvProd
        
        #Increase sum of output of BT  the coeef_add_bt if coef_add_bt_dist
        upscaled_prod_bt_total_df = prod_lv2upscale_df + coef_add_bt*prod_bt_total_1mw_df  

        
    upscaled_prod_bt_total_df.columns = ['Prod_BT']
    
    return upscaled_prodHT_P0100_df, upscaled_prod_bt_total_df




def robustControl(df_out_block_pf_opf:pandas.core.frame.DataFrame ,
                  df_hvProd_noControl: pandas.core.frame.DataFrame,
                  cur_hvProd_max: float, 
                  ctrld_hvProd_max: int , 
                  vm_mu_max: float):
    
    """
    Implement Robust control by letting the controlled Hv Producer inject all its production 
    when no voltage rise above the predefined threshold is detected. Replacement is done in
    place i.e. in the df_out_block_pf_opf 
    
    Parameters
    -----------
    df_out_block_pf_opf: (Dataframe)
        Output of the block pf opf
    df_hvProd_noControl : Dataframe
        Dataframe of P0100 with no control
    cur_hvProd_max: float
        Current Value of maximum output Power of the HV producer (MW)
    ctrld_hvProd_max: (int)
        Maximum fixed output of the Controlled Higher voltage producer (MW)
    vm_mu_max (Float)
        Threshold of voltage rise authorised
    """
    
    # Basically, we replace the value of the controled HvProd by its own 
    # value with No control when no voltage rise above the defined threshold is detected.
    
    # create new period index mask spaning from 08Am to 6PM
    per_index2 = df_out_block_pf_opf.index.to_timestamp().to_series().between_time('07:10',
                                                                         '18:50').index.to_period('10T')
    ctrld_hvProdName = df_out_block_pf_opf.columns[0]
    
    # Create a new df for hvProd
    hvProd_robust_df = pd.DataFrame(index=per_index2, columns=['hvProd_robust'])
    
     # Get into the dataframe data of hvProdWhen there is no control
    hvProd_robust_df.loc[per_index2,['hvProd_robust'] ] = (df_hvProd_noControl.loc[per_index2].P0100
                                                                  *cur_hvProd_max/ctrld_hvProd_max)
    
    # Get a mask for the periods where a voltage rise above the threshold is predicted 
    mask_vrise_per = df_out_block_pf_opf.loc[per_index2, 'max_vm_pu_pf']>vm_mu_max
       
    # Replace the values of periods given by the mask by the value of hvProd given by the persistence model
    hvProd_robust_df[mask_vrise_per] = df_out_block_pf_opf.loc[per_index2].loc[mask_vrise_per,[ctrld_hvProdName]]

    # Replace the values of hvProdin df_out_block_pf_opf
    df_out_block_pf_opf.loc[per_index2, [ctrld_hvProdName]] = hvProd_robust_df.loc[per_index2, 'hvProd_robust']  


    
    
    
def block_prod(df_out_block_pf_opf: pandas.core.frame.DataFrame, 
               df_hvProd_noControl: pandas.core.frame.DataFrame, 
               cur_hvProd_max: float, 
               ctrld_hvProd_max: int, 
               starting_index: int = 0 ):
    """
    Implement bloc prod i.e. make sure the controlled HV producer can't inject into the lower network 
    more than its actual production. Modify in place the input dataframe df_out_block_pf_opf
    
    Parameters
    ----------
    df_out_block_pf_opf: (Dataframe)
        Output of the block pf opf that has been send to the robust persitence
    df_hvProd_noControl : Dataframe
        Dataframe of hvProdwith no control
    cur_hvProd_max: int
        Value of maximum output Power of the HV producer (MW)
    ctrld_hvProd_max: (int)
        Maximum fixed output of the Controlled Higher voltage producer (MW)
    starting_index: Starting index (optional), default to zero
        Important to use this starting index and set it to the lenght of a day in the case 
        of the RNN. This is due to the fact that the prediction needs a whole day of data 
        to be produced. Especially the first prediction must be that of the the first index
        of the second day of the testing set since the whole first day (of the testing set)
        data is used .    
    
    """
    ctrld_hvProdName = df_out_block_pf_opf.columns[0]
    
    
    per_index2 = df_out_block_pf_opf.index.to_timestamp().to_series().between_time('07:10',
                                                                                   '18:50').index.to_period('10T')
    
    df_hvProd_noControl_upscaled = (df_hvProd_noControl.loc[per_index2[starting_index:],ctrld_hvProdName]
                                    *cur_hvProd_max/ctrld_hvProd_max)
    
    df_P0100_controled = df_out_block_pf_opf.loc[per_index2[starting_index:], ctrld_hvProdName]
    
    df_out_block_pf_opf.loc[per_index2[starting_index:], [ctrld_hvProdName]] = (np.minimum(df_hvProd_noControl_upscaled,
                                                                                  df_P0100_controled)
                                                                               )
        

        
        

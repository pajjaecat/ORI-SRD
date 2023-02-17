Description of Functions
=========================
We provide a succint list of the Main functions.



block_prod
***********

.. autofunction:: oriFunctions.block_prod



check_bus_connection
**********************

.. autofunction:: oriFunctions.check_bus_connection



createDict_prodHtBt_Load
*************************

.. autofunction:: oriFunctions.createDict_prodHtBt_Load



extractParRes_asDf
*******************

.. autofunction:: oriFunctions.extractParRes_asDf




improve_persinstence
********************

.. autofunction:: oriFunctions.improve_persinstence



initLowerNet_at
****************

.. autofunction:: oriFunctions.initLowerNet_at



max_vm_pu_at
*************

.. autofunction:: oriFunctions.max_vm_pu_at




predictionBin_bloc
*******************

.. autofunction:: oriFunctions.predictionBin_bloc



prediction_bloc
****************

.. autofunction:: oriFunctions.prediction_bloc



readAndReshape_excelFile
*************************

.. autofunction:: oriFunctions.readAndReshape_excelFile



robustControl
*************

.. autofunction:: oriFunctions.robustControl



robustPred
***********

.. autofunction:: oriFunctions.robustPred



run_powerflow
***************

.. autofunction:: oriFunctions.run_powerflow



run_powerflow_at
*****************

.. autofunction:: oriFunctions.run_powerflow_at



setNetwork_params
******************

.. autofunction:: oriFunctions.setNetwork_params


par_block_pfOpf
****************
Block PF/OPF using parallels engines.

Execute a power flow, an optimal power flow or both depending on ``opf_status`` that is  extracted from the parallel
engine object ``par_engines``.  If ``opf_status`` is "Both", the function is used as the block PF/OPF. If ``opf_status``
is  ``False``, the function is used as the block PF.


Due to the fact that this function uses an `Ipython<https://ipython.org/>`_ magic function, it cannot be added to a module. The function **MUST**
therefore be implemented in the local space of each notebook that make use of it as done in
`VoltageRiseBinaryUpdated<https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/VoltageRiseBinaryUpdated.ipynb>`_.


.. codeblock:: python
    def par_block_pfOpf(par_engines,
                    pred_model_f=None
                   ):
    """ Block PF/OPF using parallels engines.

    Execute a power flow, an optimal power flow or both depending on ``opf_status`` that is
    extracted from the parallel engine object ``par_engines``.  If ``opf_status`` is "Both",
    the function is used as the block PF/OPF. If ``opf_status`` is  ``False``, the function
    is used as the block PF.

    Parameters
    ----------
    par_engines: :py:class:`oriClass.CreateParEngines`
        Parallel engines object.
    pred_model_f: str, Optional, Default =  None
        'Pers' ==> Using persistence model

    Returns
    -------
    pandas.DataFrame
        ``opf_status`` is "Both":
            Result at the output of the block get_results_asDf()
        ``opf_status`` is False:
            The maximum voltage rise observed on the network after the control is implemented.

    Warnings
    ---------
    DO NOT CALL this function from the module :py:mod:`oriFunctions`. This function **Must**
    be implemented in the local space of each notebook that use it as done in
    `VoltageRiseBinaryUpdated<https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/VoltageRiseBinaryUpdated.ipynb>`_

    """



    # All  the variables used in the parallel running MUST be already sent to the local space of each engine.
    if pred_model_f == 'Pers':
        # Run problem in parallel
        %px par_run_Results = [par_oriFunctions.run_powerflow_at(lowerNet, cur_period+1, lowerNet_hv_activated_bus, sum_max_main_network,  dict_df_sgenLoad, vm_mu_max, opf_status, pred_model) for cur_period in period_part]
    else:
        %px par_run_Results = [par_oriFunctions.run_powerflow_at(lowerNet, cur_period, lowerNet_hv_activated_bus, sum_max_main_network,  dict_df_sgenLoad, vm_mu_max, opf_status) for cur_period in period_part]

    # Gather the results of all the engines in a unique variable.
    results = par_engines.gather_results('par_run_Results')

    # Wait 2seconds time for gathering the results of parralel computing.
    # This waiting time could be reduce when using more powerful machines.
    time.sleep(2)

    # return depends on opf_status defined in parallel engines.
    if par_engines.get_opf_status() == 'Both':
        extracted_results = par_engines.get_results_asDf() # Extract results
        return extracted_results

    elif par_engines.get_opf_status() == False:
        # Put data in pandas.DataFrame
        max_vm_pu_rnn_df = pd.DataFrame(data=np.array(results),
                                        index=par_engines.get_run_periodIndex(), columns=['vm_pu_max_pf'],)
        return max_vm_pu_rnn_df
    else:
        pass


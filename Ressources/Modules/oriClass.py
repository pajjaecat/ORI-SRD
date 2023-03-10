# -*- coding: utf-8 -*-

# Copyright (c) 2022-2023 by Jesse-James PRINCE AGBODJAN for SRD-Energies (https://www.srd-energies.fr/) 
# All rights reserved.

"""
OriClass - Python library with List of all classes used in the
`tutorials <https://github.com/pajjaecat/ORI-SRD/tree/main/Ressources/Notebooks>`_



"""
import os
import ipyparallel as ipp
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandapower as pp
import pandas as pd
import seaborn as sbn

import checker

from oriVariables import (defAuth_hvBus_vRiseMax,
                          defAuth_hvBus_vRiseMin,
                          default_hv_voltage,
                          default_lv_voltage,
                          Δt,
                          modules_folder)


# ------------------------------------------------------------------------------------
##############################         Class          #########################################
class CreateParEngines:
    """Create an instance of ``n_e`` parallel engines.

    Parameters
    ----------
    n_e : int, Optional, Default = 2
        The number of parallel engines to create.

    """

    def __init__(self, n_e=2):
        """ Create an instance of parallel engines.

        Parameters
        ----------
        n_e : int, Optional, default = 2
            The number of parallel engines to create.


        """
        cluster = ipp.Cluster(n=n_e)  # Create cluster
        cluster.start_cluster_sync()  # Start cluster
        self._rc = cluster.connect_client_sync()  # connect client
        self._rc.wait_for_engines(n_e)  # Wait for engines to be available for all clients
        self.dview = self._rc[:]  # Create a direct view of all engine
        """ ipp.DirectView : Direct view for parallel Engines."""

    def sendVar_to_localSpace(self,
                              run_periodIndex,
                              opf_status,
                              dict_df_sgenLoad,
                              parameters_dict,
                              clean: bool = True):
        """Send variables to the local space of each parallel engine.


        Parameters
        ----------
        run_periodIndex : pandas.PeriodIndex
            Total number of periods to run simulation for. The number of period each
            engine will work on is therefore given by ``len(run_periodIndex)/n_e``
            where ``n_e`` is the number of engines.
        opf_status : bool or str
            Optimal power flow status. Whether the maximum voltage rise on the
            ``lowerNet`` HV buses is extracted after a power flow, an optimal power
            flow  or both. Two values are possible:
                ``opf_status`` = False :
                    Run a simple power flow i.e., `pandapower.runpp(network)`
                ``opf_status`` = "Both" :
                    A power flow is run. Only when the result i.e. the voltage rise
                    detected on hv Prod
                    Buses ``max_vm_pu`` > :py:data:`oriVariables.defAuth_hvBus_vRiseMax`,
                    is the  optimal  power flow run.
        dict_df_sgenLoad : dict
            Output of :py:func:`oriFunctions.createDict_prodHtBt_Load`. See the said function for more.
        parameters_dict : dict
            TOWRITE
        clean : bool, Optional
            Whether the local space of the parallel engines should be clean or not.
                True :
                    Clear all the variables in the local space of each engine and
                    reload all needed packages.
                False :
                    Keep all the variable and packages in the local space of each
                    engine.

        Raises
        ------
        ValueErrorException :
            If ``opf_status`` or ``clean`` are the wrong type or have wrong value.


        """

        checker.check_opf_status(opf_status)  # Raise Error if the OPF type is not well defined
        checker.check_clean(clean)  # Raise error if clean Not a bool

        # Set variables
        self._run_periodIndex = run_periodIndex
        self._opf_status = opf_status

        # If `pred_model` is not present in the parameters dict
        # initialise 'self._pred_model' to None
        try:
            self._pred_model = parameters_dict['pred_model']
        except:
            self._pred_model = None

        if clean:  # Clear the localspace of all engines if clean is True and reload modules
            self.dview.clear()
            with self._rc[:].sync_imports():
                import numpy
                import pandapower
                import pandas
                import sys

            # Add modules location to Par engines namespace
            self.dview[f'sys.path.append("{modules_folder}")']

            with self._rc[:].sync_imports():  # import from modules_folder
                import oriFunctions

        # Share the total number of period in df_prodHT.index among all the created engines
        self.dview.scatter('period_part', run_periodIndex)

        # Add all variables in the parameters_dict  to  the local space of each client
        self.dview.push(parameters_dict)

        # Send following Variables to local space of parallel engines
        self.dview['opf_status'] = opf_status
        self.dview['dict_df_sgenLoad'] = dict_df_sgenLoad

    def gather_results(self, par_result_name: str):
        """ Gather in one variable the results of the parallel engines.


        Parameters
        ----------
        par_result_name : str
            Name use to call the parallel running in ``block pf_opf``.

        Returns
        -------
        gathered_results : ipyparallel.AsyncMapResult
            Result of all parallel engine.

        Raises
        ------
        ValueErrorException
            If ``par_result_name`` is not a `str`.

        See Also
        --------
        get_results_asDf
            Get parallel engines results' as a pandas.Dataframe.

        """
        # Check if the given argument is a string
        checker.check_parResultsNames(par_result_name)  # Check if the given argument is a string
        self._gathered_results = self.dview.gather(par_result_name)

        return self._gathered_results

    def get_results_asDf(self, gathered_results=None):
        """ Transform the results of the parallel engines  in a dataframe.

        Parameters
        ----------
        gathered_results: ipyparallel.AsyncMapResult, Optional, Default=None
            Output of `gather_results`. If the parameter is not given (=None) the function get the parallel engine
            results to transform from the last call of `gather_results`.




        Returns
        -------
        df : pandas.Dataframe
            The returned dataframe depends on the ``opf_status`` send to the parallel
            engines via :py:func:`sendVar_to_localSpace`. It's index are the instants
            associated with the  recorded variables in the columns which are the
            following :

            ``opf_status`` = True or "Both" :
                The df columns are
                    max_vm_pu_pf :
                        Using power flow, maximum voltage recorded over all the bus
                        at the instants given  by the ``df.index``
                    max_vm_pu :
                        Using optimal power flow, maximum voltage recorded over all
                        the bus at the instants given  by the ``df.index``. Note that
                        ``max_vm_pu = max_vm_pu_pf`` when the voltage rise constraint
                        ``max_vm_pu`` < :py:data:`oriVariables.defAuth_hvBus_vRiseMax`
                        holds.
                    [P0xxa, P0xxb, ..., P0..n] :
                        The injected power of the respective HV producers.
                    SumLv :
                        Sum of the injected power of all lower voltage producers.

            ``opf_status`` = False :
                The df columns is:
                    vm_pu_max_pf
                        Using power flow, maximum voltage recorded over all the bus
                        at the instants given by the ``df.index``

        See Also
        --------
        gather_results
            Gather in one variable the result of the parallel engines.

        Warnings
        --------
        Make sure to run :py:func:`gather_results` before :py:func:`get_results_asDf`

        """

        # Collect the parallel results from gather_results(*args)
        parallel_result = self._gathered_results

        if self._opf_status:  # If the opf is True or "Both" ----------------------------------------------

            # Get df_prodHT columns name [] from one of the engines
            df_prodHT_colName = self.dview['dict_df_sgenLoad'][-1]['df_prodHT'].columns

            # Get all the elements from the parallel result in a list
            # elm[0]      : Maximum voltage on all the line
            # elm[1]      : Power injected into the network by all the Sgen in the network
            # elm[1][0][0]: Power injected into the network by the first HV producer
            # ...
            # elm[1][0][n]: Power injected into the network by the last HV producer i.e. P0100
            # elm[1][1][0]: Power injected into the network by the first LV producer
            # ...
            # elm[1][1][n]: Power injected into the network by the Last LV producer
            # elm[2]      : Period index associated to all the previous output variable

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
                colName = ['max_vm_pu_pf', 'max_vm_pu'] + df_prodHT_colName.to_list() + SumLv_colName
            else:
                sep_list = [(elm[0], *elm[1][0], np.array(elm[1][1]).sum(), elm[2])
                            for elm in parallel_result]
                # Create a columns using 'vm_pu_max' and add the HT producers name
                colName = ['max_vm_pu'] + df_prodHT_colName.to_list() + SumLv_colName

            # Create a data based on all the cols of sep_list except the last one that is the index
            data_input = np.array(np.array(sep_list)[:, :-1], dtype=float)
            index_list = np.array(sep_list)[:, -1]  # extract the last col that is the index

        else:  # Opf status is False  ------------------------------------------------------------------

            # Get all the elements from the parallel result in a list
            # elm[0]   : Maximum voltage using power flow
            # elm[1]   : Period index associated with elm[0]
            sep_list = [(elm[0], elm[1]) for elm in parallel_result]

            # Create a data input and index
            data_input = np.array(np.array(sep_list)[:, 0], dtype=float)
            index_list = np.array(sep_list)[:, 1]

            colName = ['max_vm_pu_pf']  # column's name

        # create Output dataframe  new  dataframe based on previous unpack data
        df = pd.DataFrame(data=data_input,
                          index=index_list,
                          columns=colName)

        return df.sort_index()  # return the newly create dataFrame with the index sorted

    def _dview(self):
        return self.dview

    def get_dview(self):
        """Return a direct view into the parallel engines. """
        return self._dview()

    def get_run_periodIndex(self):
        """Return  the total number of periods for which the simulation is done."""
        return self._run_periodIndex

    def get_pred_model(self):
        """Return the prediction model."""
        return self._pred_model

    def get_opf_status(self):
        """Return the optimal power flow status"""
        return self._opf_status


class InitNetworks:
    """ Initialize both the upper and lower level  Networks.


    Parameters
    ----------
    upperNet : :obj:`pandapower.pandapowerNet`
        Upper level network.
    lowerNet : `pandapower.pandapowerNet `
        Lower level Network
    coef_add_bt : float, Optional, Default = None
        Value of the added output power for all the LV producers (MW).
    coef_add_bt_dist : str, Optional, Default = None
        How the upscaling of the maximum output of all lower Voltage producers is
        done. Three choices are possible
            None
                No upscaling is done.
            "uppNet"
                ``coef_add_bt`` is added to the Sum of maximum output of all
                lower   voltage (LV) producers (MW) in the upper Network. In
                consequence, the LV producers on the ``lowerNet`` receive only a
                fraction of coef_add_bt.
            "lowNet"
                ``coef_add_bt`` is added to the Sum of maximum output of all LV
                producers (MW) in the ``lowerNet``. In consequence, coef_add_bt
                is shared proportionally among all the LV producers on the
                ``lowerNet``.
            "lowNet_rand"
                ``coef_add_bt`` is shared proportionally among a randomly
                selected et of the LV  producers on the ``lowerNet``. The
                randomly selected set consist  of half of all LV producers
                on the ``lowerNet``

    Attributes
    ----------

    Raises
    ------
    Exception
        If ``upperNet`` has fewer buses that ``lowerNet`` or if ``coef_add_bt`` and
        ``coef_add_bt_dist`` are the wrong type.


    """

    def __init__(self,
                 upperNet,
                 lowerNet,
                 coef_add_bt: float = None,
                 coef_add_bt_dist: str = None):
        """ Initialize both the upper and lower level  Networks

        Parameters
        ----------
        upperNet : :obj:`pandapower.pandapowerNet`
            Upper level network.
        lowerNet : pandapower.pandapowerNet
            Lower level Network
        coef_add_bt : float, Default = None
            Value of the added output power for all the LV producers (MW).
        coef_add_bt_dist : str, Default = None
            How the upscaling of the maximum output of all lower Voltage producers is
            done. Three choices are possible
                None
                    No upscaling is done
                "uppNet"
                    ``coef_add_bt`` is added to the Sum of maximum output of all
                    lower voltage (LV) producers (MW) in the upper Network. In
                    consequence, the LV producers on the ``lowerNet`` receive only a
                    fraction of coef_add_bt.
                "lowNet"
                    ``coef_add_bt`` is added to the Sum of maximum output of all LV
                    producers (MW) in the ``lowerNet``. In consequence, coef_add_bt
                    is shared proportionally among all the LV producers on the
                    ``lowerNet``.
                "lowNet_rand"
                    ``coef_add_bt`` is shared proportionally among a randomly
                    selected set of the LV  producers on the ``lowerNet``. The
                    randomly selected set consists of half of all LV producers
                    on the ``lowerNet``

         """

        # initiate private attribute
        self._upperNet = upperNet
        self._lowerNet = lowerNet
        self._coef_add_bt = coef_add_bt
        self._coef_add_bt_dist = coef_add_bt_dist

        # Checking
        checker.check_network_order(self)
        checker.check_coef_add_bt_and_dist(self)

        # Extract LowerVoltage Producers # Their names is None i.e. nan is the _lowerNet.sgen
        self._lowerNet_sgenLvCopy = self._lowerNet.sgen[self._lowerNet.sgen.name.isna()]

        # Get sum of maximum output of all the LV producers on the lower network before update
        self._lowerNet_nonUpdated_sum_max_lvProd = self._lowerNet_sgenLvCopy.max_p_mw.sum()

        self.lowerNet_update_max_p_mw()  # Update the maximum output of LV producers given _coef_add_bt and
        # _coef_add_bt_dist

    def lowerNet_update_max_p_mw(self):
        """ Update or upscale the maximum output of the all LV producers on the ``lowerNet``.

        The upscaling done depends on ``coef_add_bt`` and ``coef_add_bt_dist``
        """

        if self._coef_add_bt_dist == 'lowNet':  #
            updated_lowerNet_sum_max_lvProd = self._lowerNet_nonUpdated_sum_max_lvProd + self._coef_add_bt
            self._lowerNet.sgen.max_p_mw[self._lowerNet.sgen.name.isna()] = (updated_lowerNet_sum_max_lvProd
                                                                             * self._lowerNet_sgenLvCopy.max_p_mw
                                                                             / self._lowerNet_nonUpdated_sum_max_lvProd)
        elif self._coef_add_bt_dist == 'lowNet_rand':
            # -----------------------              TODO: CODE                ------------------------------
            pass
        elif self._coef_add_bt_dist == 'uppNet':
            # In this case, the The upscaling is done in oriFunctions.upscale_HvLv_prod(*args)
            pass

    def _upperNet_sum_max_lvProdLoad(self):
        """ On the upper Network, compute and return sum(LvProd_max) and sum(LvProd_max).

        These are respectively the (1) sum of maximum output of all LV producers and
        (2) maximum of all Load demand.
        """
        sum_max_lvProd = self._upperNet.sgen[self._upperNet.sgen.name.isna()].max_p_mw.sum()
        sum_max_load = self._upperNet.load.max_p_mw.sum()

        return sum_max_lvProd, sum_max_load

    def _lowerNet_sum_max_lvProdLoad(self):
        """ On the ``lowerNet``, compute and return sum(LvProd_max) and sum(LvProd_max).

        These are respectively the (1) sum of maximum output of all LV producers and
        (2) maximum of all Load demand.

        """
        sum_max_lvProd = self._lowerNet.sgen[self._lowerNet.sgen.name.isna()].max_p_mw.sum()
        sum_max_load = self._lowerNet.load.max_p_mw.sum()

        return sum_max_lvProd, sum_max_load

    def _lowerNet_hv_bus_df(self,
                            hvBus_voltage):
        """ Extract the higher Voltage buses in the ``lowerNet``: These are buses for which
            the parameter \'hvBus_voltage\' equals 20.6 kV.
        """
        return self._lowerNet.bus.groupby('vn_kv').get_group(hvBus_voltage)

    def _lowerNet_lv_bus_df(self,
                            lvBus_voltage):
        """ Extract the lower Voltage buses in the ``lowerNet``: These are buses for which
            the parameter \'lvBus_voltage\' equals 0.4 kV.
        """
        return self._lowerNet.bus.groupby('vn_kv').get_group(lvBus_voltage)

    def _run_lowerNet(self):
        """ Run ``lowerNet`` """
        pp.runpp(self._lowerNet)

    def lowerNet_set_vrise_threshold(self,
                                     lowNet_activatedBus_list: list,
                                     min_vm_mu: float = defAuth_hvBus_vRiseMin,
                                     max_vm_mu: float = defAuth_hvBus_vRiseMax):
        """ Set min and max voltage rise threshold on the ``lowerNet``.

        The minimum and the maximum  authorised thresholds  are set on  the list of
        buses  given by ``lowNet_activatedBus_list`` on the ``lowerNet``.

        This method can be utilised to set the authorised voltage rise on both the
        higher and  lower voltage Buses.  If the desire is to set the authorised
        voltage rise on the hv buses, ``lowNet_activatedBus_list`` **MUST** be the
        list of lowNet_hv_activatedBus. If the desire is to set the authorised
        voltage rise on the lv buses, parameter `lowNet_activatedBus_list`` **MUST**
        be the list  of lowNet_lv_activatedBus.


        Parameters
        ----------
        lowNet_activatedBus_list : list
            List of all the activated buses on the ``lowerNet``
        min_vm_mu : float, Default = :py:data:`oriVariables.defAuth_hvBus_vRiseMin`
            Minimum authorised voltage rise on ``lowNet_activatedBus_list``.
        max_vm_mu : float, Default = :py:data:`oriVariables.defAuth_hvBus_vRiseMax`
            Maximum authorised voltage rise on ``lowNet_activatedBus_list``.

        """

        self._lowerNet.bus.max_vm_pu[lowNet_activatedBus_list] = max_vm_mu
        self._lowerNet.bus.min_vm_pu[lowNet_activatedBus_list] = min_vm_mu

    def init_controlled_hvProd(self,
                               controlled_hvProdName: str):
        """ Initialize the controlled higher and lower voltage producers in the ``lowerNet``.

        The initialization implies adding a <<controllable>> column to the sgen's in
        the ``lowerNet`` and setting the controlled producers to ``True``.

        Parameters
        ----------
        controlled_hvProdName : str
            Name of the controlled HV producer on the ``lowerNet``

        Raises
        ------
        Exception
            If ``controlled_hvProdName`` is not present in the list of HV producer in
            the ``lowerNet``.
        """

        # Check the existence of the controlled_hvProdName
        checker.check_hvProdName_in_lowerNet(self, controlled_hvProdName)

        # Add a controllable line to the static generators
        self._lowerNet.sgen['controllable'] = False

        # Set the controlled HV producer as a controllable sgen
        self._lowerNet.sgen['controllable'][self._lowerNet.sgen.name == controlled_hvProdName] = True

        # Set all the LV producers as controllable depending on self._coef_add_bt_dist
        if self._coef_add_bt_dist in ['lowNet', 'lowNet_rand']:
            self._lowerNet.sgen['controllable'][self._lowerNet.sgen.name.isna()] = True

        # Add Missing columns to be able to run an opf
        self._lowerNet.sgen[['min_p_mw', 'min_q_mvar', 'max_q_mvar']] = 0.

        # Rename bus parameters because the names do not correspond to the
        # parameters in pandapower
        self._lowerNet.bus.rename({'max_vw_pu': 'max_vm_pu'}, axis=1, inplace=True)

        # Delete useless parameters
        self._lowerNet.bus.drop(['max_vm', 'min_vm'], axis=1, inplace=True)

    def get_lowerNet_hvActivatedBuses(self,
                                      lowerNet_hvBuses_list: list):
        """ Return a list of all HV activated buses (vn_kv=20.6) on the ``lowerNet``.

        Parameters
        ----------
        lowerNet_hvBuses_list : list
            List of all hv buses in the ``lowerNet``.

        Returns
        -------
        list
            List of hv Activated Buses on the ``lowerNet``.


        """

        # Run a power flow on the lower net only if it is not run already.
        if not checker.check_resTables_existing(self):
            self._run_lowerNet()

        # Extract a list of  all the activated bus on the run lower network
        lowerNet_activatedBuses_index = list(self._lowerNet.res_bus.vm_pu[
                                                 self._lowerNet.res_bus.vm_pu.notna()].index)

        # Extract the list of all HV activated Bus on the lower network
        self._lowerNet_hvActivatedBuses_list = [bus_index for bus_index in lowerNet_activatedBuses_index
                                                if bus_index in lowerNet_hvBuses_list]
        return self._lowerNet_hvActivatedBuses_list

    def get_lowerNet_lvActivatedBuses(self,
                                      lowerNet_lvBuses_list: list) -> list:
        """ Return a list of all LV activated buses (vn_kv=0.4) on the ``lowerNet``

        Parameters
        ----------
        lowerNet_lvBuses_list : list
            List of all lv buses in the ``lowerNet``.

        Returns
        -------
        list
            List of Lv Activated Buses on the ``lowerNet``.

        """

        # Run a power flow on the lower net only if it is not run already.
        if not checker.check_resTables_existing(self):
            self._run_lowerNet()

        # Extract a list of  all the activated bus on the run lower network
        lowerNet_activatedBuses_index = list(self._lowerNet.res_bus.vm_pu[
                                                 self._lowerNet.res_bus.vm_pu.notna()].index)

        # Extract the list of all HV activated Bus on the lower network
        self._lowerNet_lvActivatedBuses_list = [bus_index for bus_index in lowerNet_activatedBuses_index
                                                if bus_index in lowerNet_lvBuses_list]
        return self._lowerNet_lvActivatedBuses_list

    def _get_lowerNet_hvProducersName_df(self):
        """Return the name of all the Higher voltage producers on the lower net as a dataframe."""
        return self._lowerNet.sgen.name[self._lowerNet.sgen.name.notna()]

    def _get_lowerNet_lvProducersName_df(self):
        """Return the name of all the lower voltage producers on the lower net as a dataframe."""
        return self._lowerNet.sgen.name[self._lowerNet.sgen.name.isna()]

    def get_lowerNet_hvProducersName(self,
                                     return_index=False):
        """Return the name (and if needed the index) of all the Higher voltage producers on the lower net.

        Parameters
        ----------
        return_index : bool, Optional, Default = False
            False
                The indexes of the HV producers on sgen table are not returned.
            True
                The indexes of the HV producers on sgen table are returned along with
                the HV producer's name.

        """

        if return_index:
            return (list(self._get_lowerNet_hvProducersName_df().values),
                    list(self._get_lowerNet_hvProducersName_df().index))
        else:
            return list(self._get_lowerNet_hvProducersName_df().values)

    def get_lowerNet_lvProducersName(self, return_index=False):
        """Return the name of all the Lower voltage producers on the lower net.

        Parameters
        ----------
        return_index : bool, optional, Default = False
            False
                The indexes of the LV producers on sgen table are not returned.
            True
                The indexes of the LV producers on sgen table are returned along with
                the HV producer's name.

        """

        if return_index:
            return (list(self._get_lowerNet_lvProducersName_df().values),
                    list(self._get_lowerNet_lvProducersName_df().index))
        else:
            return list(self._get_lowerNet_lvProducersName_df().values)

    def get_params_coef_add_bt(self) -> tuple:
        """Return the parameters ``coef_add_bt`` and ``coef_add_bt_dist``. """
        return self._coef_add_bt, self._coef_add_bt_dist

    def get_ctrld_hvProdName(self,
                             return_index=False):
        """Return the name (and if needed the index) of the Controlled HV producer on the lower net.

        Parameters
        ----------
        return_index : bool, optional, Default = False
            False
                The index of the HV producer on sgen table are not returned.
            True
                The index of the HV producer on sgen table are returned along with
                the HV producer's name.

        """
        mask_hv_prod = self._lowerNet.sgen.name.notna()
        hv_prod_df = self._lowerNet.sgen[mask_hv_prod]

        if return_index:
            return (list(hv_prod_df[hv_prod_df.controllable].name)[0],
                    list(hv_prod_df[hv_prod_df.controllable].index)[0])
        else:
            return list(hv_prod_df[hv_prod_df.controllable].name)[0]

    def get_ctrld_lvProdName(self,
                             return_index=False):
        """Return the name (and if needed the index) of the Controlled LV producers on the lower net.

        Parameters
        ----------
        return_index : bool, optional, Default = False
            False
                The index of the LV producer on sgen table are not returned.
            True
                The index of the LV producer on sgen table are returned along with
                the HV producer's name.

        """
        mask_hv_prod = self._lowerNet.sgen.name.isna()
        hv_prod_df = self._lowerNet.sgen[mask_hv_prod]

        if return_index:
            return (list(hv_prod_df[hv_prod_df.controllable].name),
                    list(hv_prod_df[hv_prod_df.controllable].index))
        else:
            return list(hv_prod_df[hv_prod_df.controllable].name)

    def get_upperNet(self):
        """Return the Upper network."""
        return self._upperNet

    def get_lowerNet(self):
        """Return the ``lowerNet``."""
        return self._lowerNet

    def get_upperNet_sum_max_lvProdLoad(self):
        """On the upper network, compute and return sum(LvProd_max) and sum(LvProd_max).

        These are respectively the (1) sum of maximum output of all LV producers and
        (2) maximum of all Load demand.

        """
        return self._upperNet_sum_max_lvProdLoad()

    def get_lowerNet_sum_max_lvProdLoad(self):
        """On the ``lowerNet``, compute and return sum(LvProd_max) and sum(LvProd_max).

        These are respectively the (1) sum of maximum output of all LV producers and
        (2) maximum of all Load demand.

        """
        return self._lowerNet_sum_max_lvProdLoad()

    def get_lowerNet_hv_bus_df(self,
                               hvBus_voltage: float = default_hv_voltage):
        """ Return a list of the higher Voltage buses (vn_kv=20.6 kV) in the ``lowerNet``.


        Parameters
        ----------
        hvBus_voltage : float, Optional, Default = ``default_hv_voltage``
            Voltage of High voltage buses on ``lowerNet``.

        """
        return self._lowerNet_hv_bus_df(hvBus_voltage)

    def get_lowerNet_lv_bus_df(self,
                               lvBus_voltage: float = default_lv_voltage):
        """ Return a list of the lower Voltage buses (vn_kv=0.4 kV) in the ``lowerNet``.


        Parameters
        ----------
        lvBus_voltage : float, Optional, Default = :py:data:`oriVariables.default_lv_voltage`
            Voltage of lower voltage buses on ``lowerNet``.

        """
        return self._lowerNet_lv_bus_df(lvBus_voltage)

    def create_lowerNet_sgenDf_copy(self):
        """Create a copy of ALL the Producer in the ``lowerNet``"""
        self._lowerNet_sgenDf_copy = self._lowerNet.sgen.copy(deep=True)

    def get_lowerNet_sgenDf_copy(self):
        """Return a copy of the initial sgen (both the Lv and HV Prod) on the ``lowerNet``"""
        # Run create_lowerNet_sgenDf_copy to create the _lowerNet_sgenDf_copy
        self.create_lowerNet_sgenDf_copy()

        return self._lowerNet_sgenDf_copy  # return the created copy


class SensAnalysisResult:
    """
    Initiate the Sensitivity analysis with the folder_location.

    Parameters
    ----------
    folder_location : str
        Location of the folder where the results of the sensitivity analysis are
        stored.

    """

    def __init__(self, folder_location):
        self._folder_location = folder_location
        self._files_in_folder_list = os.listdir(self._folder_location)
        self._plkFiles_in_folder_list = self._extractPlkFiles(self._files_in_folder_list)
        self._check_fileName_Format(self._plkFiles_in_folder_list)
        self._sortedPlkFiles_in_folder_list = self._sort_plkFiles_in_folder(self._plkFiles_in_folder_list)

    def _check_fileName_Format(self,
                               plkFiles_in_folder_list):
        # Check if the files name are in the expected format. The Expected format ought to be
        # modelName_btRangeName_SimNumber.plk such that the split length ==3
        len_splited = len(plkFiles_in_folder_list[0].split('_'))

        if len_splited > 3:
            raise Exception('The *.plk files in ' + self._folder_location + 'are not in the expected format. \n'
                            + '\t   Make sure they are named such as modelName_btRangeName_SimNumber.plk')

    def _extractPlkFiles(self,
                         files_in_folder_list):
        # Extract only the plk files in folderlocation
        plk_files_list = [cur_file for cur_file in files_in_folder_list if cur_file.endswith('.plk')]
        return plk_files_list

    def _sort_plkFiles_in_folder(self,
                                 plkFiles_in_folder_list):

        first_plkFile_in_folder_name = plkFiles_in_folder_list[0]

        # cur_file.split('_')[-1] is used to get the the last elm which is something like 'n.plk'
        #                                                                          where  n is an integer
        files_indexAndExtenxion_list = [cur_file.split('_')[-1] for cur_file in plkFiles_in_folder_list]

        # Extract the files index and sort them in ascending order
        # file_index.split('.')[0] is used to get the file index i.e,  || n where  n is an integer
        sorted_index = np.sort([int(file_index.split('.')[0]) for file_index in files_indexAndExtenxion_list
                                ])

        # Separate first file in folder name
        self._prediction_model_name, bt_range_name, _ = first_plkFile_in_folder_name.split('_')

        # Create a new list spaning from n to the total number of element in plkFiles_in_folder_list
        files_in_folder_list_out = [self._prediction_model_name + '_' + bt_range_name + '_' + str(elm) + '.plk'
                                    for elm in sorted_index]

        return files_in_folder_list_out

    def in_dataFrame(self,
                     start_date=None,
                     end_date=None):
        """Transform all the saved results from multivariate simulation in a dataframe.

        Each element in the resulting dataFrame is the curtailed energy for that
        particular simulation. The df's index represent the variation of the maximum
        output of the controlled HV producer while the columns represent the added BT
        production in the network.


        Parameters
        ----------
        start_date : str or pandas.Period, optional
            First day (included) to consider in the testing set for the simulation.
            If the argument `is given, the curtailed energy (each element of the df)
            considers the simulation starting from the given date. Else the whole
            data is considered
        end_date : (str) optional
            Last day (not included) to consider in the testing set for the
            simulation.
            If the argument `is given, the curtailed energy (each element of the df)
            considers the simulation up to the given date. Else the whole data is
            considered

        """

        res_dict = {}  # Dictionary to save variables

        for curFileName in self._sortedPlkFiles_in_folder_list:
            cur_file_data = joblib.load(self._folder_location + curFileName)  # Load files
            cur_file_data_keys_list = list(cur_file_data.keys())  # Get keys names in the current file
            energy_curt_list = []  # Create an energy list

            for cur_key in cur_file_data_keys_list:  # for each element in the loaded dictionary
                data_df = cur_file_data[cur_key]['Power Sgen']

                # data_df.iloc[:,0] => Power injected when there is  no control
                # data_df.iloc[:,1] => Power injected using current controller
                if end_date is None:
                    power_curt = (data_df.iloc[:, 0] - data_df.iloc[:, 1]).sum()
                else:
                    mask_period_interest = (data_df.index >= start_date) & (data_df.index <= end_date)
                    power_curt = (data_df[mask_period_interest].iloc[:, 0]
                                  - data_df[mask_period_interest].iloc[:, 1]).sum()

                energy_curt_list.append(power_curt * Δt)

            col_name = cur_file_data_keys_list[0].split()[0]
            res_dict.update({col_name: energy_curt_list})

        # Create index name for the resulting dataframe
        df_index = [key.split()[1].split('=')[1] for key in cur_file_data_keys_list]

        # Crete resulting dataframe
        self._res_df = pd.DataFrame(res_dict, index=df_index)

        #         # Rename column of  resulting dataframe
        #         self._res_df.columns = [elm.split('=')[1] for elm in self._res_df.columns]

        return self._res_df

    def print_sorted_filesNames(self):
        print(self._plkFiles_in_folder_list)

    def plot_heatmap(self, fig_params=None,
                     contour_color='yellow',
                     contour_level=np.arange(0, 700, 100),
                     colmap='twilight',
                     show_ylabel=False,
                     show_cbar=True,
                     show_contour=True,
                     anotation=False, ):

        # Check whether _res_df is already defined i.e self.in_dataFrame()
        # is already executed once. If yes, there is no exception, otherwise
        # execute the function
        try:
            getattr(self, '_res_df')
        except AttributeError:
            self._res_df = self.in_dataFrame()

        # if fig_params is not given plot the heatmap in a new figure otherwise fig_params must
        # be an axe from plt.subplots()
        # TODO Verify if fig_params is actually an axe and issue an error in the contrary
        if fig_params is None:
            fig, axx = plt.subplots(figsize=(10, 6), dpi=100)
        else:
            axx = fig_params

        x_contours = range(len(self._res_df.columns))
        y_contours = range(len(self._res_df.index))

        if show_contour:
            cntr = axx.contour(x_contours, y_contours,
                               self._res_df.iloc[::-1, :],
                               levels=contour_level,
                               colors=contour_color,
                               linewidths=1)
            # Contour lables
            axx.clabel(cntr, fmt='%1.0f', inline_spacing=10, fontsize=10)

        # colorbar kwargs
        clbar_kwargs = dict(label="Curtailed Energy (MWh/Year)", anchor=(0, .5), shrink=0.7)

        # anotation kwargs
        annot_kw = dict(size=10)

        # actual plot()
        sbn.heatmap(self._res_df.iloc[::-1, :],
                    ax=axx,
                    annot_kws=annot_kw,
                    fmt='.0f',
                    lw=0.,
                    annot=anotation,
                    cbar=show_cbar,
                    cbar_kws=clbar_kwargs,
                    cmap=colmap)

        if show_cbar & show_contour:
            # axx collection[7] contains the colorbar
            axx.collections[7].colorbar.add_lines(cntr)

        axx.set(
            xlabel='BT Production increase Variation (Mwh)',
            title=self._prediction_model_name);

        if show_ylabel:
            axx.set(ylabel='P0100 Maximum Prod (MWh)', );


class SensAnalysisResults(SensAnalysisResult):  # This class inherits super properties from the  SensAnalysisResult
    """
    Initiate the Sensitivity analysis with the folder's (location) associated
    with  each model to consider.

    """

    def __init__(self,
                 models_folder_location,
                 models_name,
                 testSet_date,
                 p_Hv_Lv_range):
        """
        Parameters
        ----------
        models_folder_location : tuple of str
            The relative folder location of Each Model to consider
        models_name : tuple of str
            The name of each model to consider
        testSet_date : (tuple of str)
            (0) test_set_starting date included
            (1) test set stopping date not includes
        p_Hv_Lv_range : (tuple of array)
            IMPORTANT : Make sure that these parameters are the same that are used in
            the  notebook Sensitivity analysis simulation
            (0) P_Hv_max_range : Range of Maximum Power for the controlled Producer
            (1) P_Lv_max_range : Range of Maximum Power added for all the Lower
            voltage producer

        """

        self._models_folder_location = models_folder_location
        self._models_name = models_name
        self._testSet_startDate = testSet_date[0]
        self._testSet_endDate = testSet_date[1]
        checker.check_input_concordance(self)
        self._files_in_folder_list_dict = {}
        self._plkFiles_in_folder_list_dict = {}
        self._sortedPlkFiles_in_folder_list_dict = {}
        checker.check_numberOf_plk_Files_in_folders(self)
        self._P0100_max_range = p_Hv_Lv_range[0]  # Define the maximum power output P0100
        self._bt_add_range = p_Hv_Lv_range[1]

        # For each model or folder do : (1) List files in folder, (2) Extract plk Files,
        #                               (3) Check files format,   (4) Sort plk files
        # and add the result in the corresponding dictionary
        for cur_model_name, cur_model_folder in zip(self._models_name, self._models_folder_location):
            self._files_in_folder_list_dict.update({cur_model_name: os.listdir(cur_model_folder)})
            self._plkFiles_in_folder_list_dict.update({cur_model_name:
                super()._extractPlkFiles(self._files_in_folder_list_dict[cur_model_name])
                                                       })
            super()._check_fileName_Format(self._plkFiles_in_folder_list_dict[cur_model_name])

            self._sortedPlkFiles_in_folder_list_dict.update({cur_model_name:
                super()._sort_plkFiles_in_folder(
                    self._plkFiles_in_folder_list_dict[cur_model_name])
            })

    def _read_files_at(self,
                       bt_file_index: int,
                       ht_file_index_list: list):
        """
        For each model_name in _sortedPlkFiles_in_folder_list_dict[model_name], read
        the file at bt_file_index to extract both :
            (1) the voltage rise dataframe ('maxV_rise_df')
            (2) the Power injected dataframe with and without control ('Power Sgen')
        for the simulation indexed by each elm in ht_file_index_list. Save each
        extracted variable in the corresponding dictionary that are output

        """

        voltage_rise_df_dict = {}
        power_df_dict = {}
        show_exception_message = True

        for cur_ht_file_index in ht_file_index_list:
            for cur_model_name, cur_model_folder in zip(self._models_name, self._models_folder_location):
                plk_file_name = self._sortedPlkFiles_in_folder_list_dict[cur_model_name][bt_file_index]
                file2read_path = cur_model_folder + plk_file_name
                bt_file_dict = joblib.load(file2read_path)  # Load file
                bt_file_dict_keys_list = list(bt_file_dict.keys())  # get keys from the dict that is
                #                the loaded file
                key2use = bt_file_dict_keys_list[cur_ht_file_index]
                name2use = cur_model_name + ' ' + key2use
                voltage_rise_df_dict.update({name2use: bt_file_dict.get(key2use)['maxV_rise_df']})

                try:  # catch exception If the column 'Power Sgen' is not present in the read dataFrame
                    power_df_dict.update({name2use: bt_file_dict.get(key2use)['Power Sgen']})
                except KeyError:
                    if show_exception_message:  # show exception message only once
                        print(f'The column [\'Power Sgen\'] is not present in',
                              f'files located in {cur_model_folder}')
                        show_exception_message = False

        return voltage_rise_df_dict, power_df_dict

    def _vRise_dict_as_dataFrame(self,
                                 voltageRise_df_dict,
                                 power_df_dict,
                                 v_rise_thresh):
        """ Transform the voltageRise_df_dict (dictionary of dataframe, each key
        being the result of a simulation) in a dataframe that will be used for
        plotting,  create var vrise_count_dict ( a dictionary of the total number of
        voltage rise above the threshold for each keys in voltageRise_df_dict) and
        var caping_count_dict  ( a dictionary of the total number of capping
        for each keys in voltageRise_df_dict)

        Parameters
        ----------
        voltageRise_df_dict: dict of dataframe
            Each key has the following format: '{model_name} BT+={bt} P0100={ht}'
            where {model_name} is self._models_name
                  {bt}         is bt argument in vRise_boxplot(**)
                  {ht}         is ht argument in vRise_boxplot(**)
        v_rise_thresh: optional, float
            Voltage rise threshold

        Returns
        -------
        df2use : dataframe
            Resulting Dataframe
        """

        df2use = pd.DataFrame(columns=['V_rise', 'Model', 'Power'])  # Create an empty dataframe with following column

        self._vrise_count_dict = {}  # Create empty dictionary
        self._capping_count_dict = {}  # Create empty dictionary

        for cur_key in voltageRise_df_dict.keys():  # For each key in voltageRise_df_dict
            # the same keys are in power_df_dict

            # Extract dataframe containing the voltage rises and injected powers
            # recorded for the selected simulation
            cur_vRise_df_model = voltageRise_df_dict[cur_key]

            # create a mask that span # only the interested period
            mask = ((cur_vRise_df_model.index >= self._testSet_startDate)
                    & (cur_vRise_df_model.index <= self._testSet_endDate))

            # filter dataframes using  mask
            cur_vRise_df_filtered = cur_vRise_df_model[mask]
            # Extract solely data where the voltage rise recorded is above the define threshold
            cur_vRise_df_filtered = cur_vRise_df_filtered[cur_vRise_df_filtered >= v_rise_thresh]
            # Update dictionaries
            self._vrise_count_dict.update({cur_key: len(cur_vRise_df_filtered.dropna())})

            # The try except is used here because for some models the cur_key is not present
            # in the power_df_dict
            try:
                cur_power_df_model = power_df_dict[cur_key]
                mask = ((cur_power_df_model.index >= self._testSet_startDate)
                        & (cur_power_df_model.index <= self._testSet_endDate))

                cur_power_df_filtered = cur_power_df_model[mask]
                self._capping_count_dict.update({cur_key:
                                                     self._compute_capping_count(cur_power_df_filtered)
                                                 })
            except KeyError:
                pass

            # Create an intermediate dataframe
            df = pd.DataFrame(cur_vRise_df_filtered.dropna().values, columns=['V_rise'])
            df[['Model', 'Power']] = cur_key.split()[0], cur_key.split()[1] + ' ' + cur_key.split()[2]
            df2use = pd.concat([df2use, df])  # Concatenate created dataframe with the existing dataframe

        return df2use

    def _compute_capping_count(self,
                               injected_power_df):
        """ Compute and return the number of capping that occurred given
        the input power dataframe """

        # injected_power_df.iloc[:,0] => Power injected when there is  no control
        # injected_power_df.iloc[:,1] => Power injected using controller

        # Create a lambda function named equality_check that will verify if for the current row,
        # the power injected with controlled On is the same as when no control is applied.
        # When both values are not equal (output True i.e. a capping or curtailment is occurring)
        #                          equal (output False i.e. a no capping)
        # equality_check = lambda cur_row: True if cur_row[1]!=cur_row[0] else False
        equality_check = lambda cur_row: cur_row[1] != cur_row[0]

        # apply equality check to each row of the df and sum the resulting df
        nb_event = (injected_power_df.apply(equality_check, axis=1)).sum()

        return nb_event

    def vRise_boxplot(self,
                      ht=0.,
                      bt=0.,
                      v_rise_thresh=1.025,
                      fig_params=None
                      ):
        """ Create a box plot of voltage rise above the defined threshold.

        This for simulations where the controlled HT producer maximum power is ``ht``
        and the added maximum bt power is arg(bt)

        Parameters
        ----------
        ht : (float, default = 0)
            Maximum Power of the controlled HT producer
        ht : (str,  'All')
           Range of all variation of the Maximum Power of the controlled HT producer
        bt : (float, default = 0)
            Added Power to the BT producers in the network
        v_rise_thresh : (float, default = 1.025)
            Maximum authorised threshold
        fig_params : (Figure axis, optional)
            Figure axis where to plot the current box plots

        """

        self._bt = bt
        self._ht = ht
        checker.check_boxplot_inputs(self, ht, bt)  # Check if boxplot inputs are authorised
        # Extract the number of the bt file to read
        bt_file_to_read_index = [file_ind for file_ind, curValue in enumerate(self._bt_add_range)
                                 if curValue == bt][0]

        if type(ht) is float:  # -----------------------   ht is a float           -----------------------
            # Extract the dict keys index of the HT simulation in the saved BT file
            ht_dict_keys_to_read_index = [file_ind for file_ind, curValue in enumerate(self._P0100_max_range)
                                          if curValue == ht][0]

            # Read file and extract variable into a dictionary
            voltageRise_df_dict, power_df_dict = self._read_files_at(bt_file_to_read_index,
                                                                     [ht_dict_keys_to_read_index])

            # convert the read dictionary in a dataframe to plot
            df_to_plot = self._vRise_dict_as_dataFrame(voltageRise_df_dict, power_df_dict, v_rise_thresh)

            if fig_params is not None:
                axx = fig_params
                if len(df_to_plot) != 0:  # Plot dataframe only if some data is present
                    sbn.boxplot(x='Power', y='V_rise', data=df_to_plot,
                                hue='Model', hue_order=list(self._models_name),
                                width=0.5, fliersize=1, linewidth=0.8, ax=axx, )
            else:
                fig, axx = plt.subplots(1, figsize=(3, 5))
                if len(df_to_plot) != 0:  # Plot dataframe only if some data is present
                    sbn.boxplot(x='Power', y='V_rise', data=df_to_plot,
                                hue='Model', hue_order=list(self._models_name),
                                width=0.5, fliersize=1, linewidth=0.8, ax=axx, )

        elif type(ht) is str:  # -----------------------  if ht is str (All)           -     ---------------
            ht_dict_keys_to_read_index = range(len(self._P0100_max_range))  #
            voltageRise_df_dict, power_df_dict = self._read_files_at(bt_file_to_read_index,
                                                                     list(ht_dict_keys_to_read_index))
            # convert the read dictionary in a dataframe to plot
            df_to_plot = self._vRise_dict_as_dataFrame(voltageRise_df_dict, power_df_dict, v_rise_thresh)

            if fig_params is None:  # define a figure is an axis is not given as input
                fig, axx = plt.subplots(1, figsize=(15, 6))
            else:
                axx = fig_params

            sbn.boxplot(x='Power', y='V_rise', data=df_to_plot,
                        hue='Model', hue_order=list(self._models_name),  # Actual Plot
                        width=0.7, fliersize=1, linewidth=0.8, ax=axx)

            # Each elm in df_to_plot.Power.unique().tolist() gives BT+=bt_max P0100=ht_max
            # Hence the elm.split()[-1]                extracts P0100=ht_max and
            #           elm.split()[-1].split('=')[-1] extracts ht_max
            ht_max_labels_list = [elm.split()[-1].split('=')[-1]
                                  for elm in df_to_plot.Power.unique().tolist()]

            axx.set_xticks(axx.get_xticks(), ht_max_labels_list)
            ticks_ = np.arange(v_rise_thresh, 1.037, 0.001)
            axx.set_yticks(ticks=ticks_,
                           labels=np.round(ticks_, decimals=3))

            axx.set_xlabel('HV Max Prod (MW)')
            axx.set_title(f'Lv Added Power = {bt} MW')

    def _count_dict_as_dataFrame(self,
                                 var_count_dict):
        """ Transform the variable indexed by var_count_dict  into a dataframe.
        Note that var_count_dict is either <<_vrise_count_dict>> or <<_capping_count_dict>>
        """

        # check the exixstence of var_count_dict in the local space of the instance
        checker.check_countDictAsDf_input_inLocalSpace(self, var_count_dict)

        # Get the self variable i.e. vRiseOrCapping_count_dict = self._vrise_count_dict
        #                         or vRiseOrCapping_count_dict = self._capping_count_dict
        vRiseOrCapping_count_dict = getattr(self, var_count_dict)

        # Extract all the ht_max in the self._vrise_count_dict to create a list
        # cur_key is such that  BT+=bt_max P0100=ht_max
        # Hence the cur_key.split()[-1]                extracts P0100=ht_max
        #       and cur_key.split()[-1].split('=')[-1] extracts ht_max that is converted in float
        list_index = [float(cur_key.split()[-1].split('=')[-1])
                      for cur_key in vRiseOrCapping_count_dict if cur_key.startswith(self._models_name[0])]

        df_to_plot = pd.DataFrame(index=list_index)  # Create an empty dataframe with the index being list_index

        for cur_mod_name in self._models_name:
            # Extract the total number of voltage rise above threshold  associated with all the element in
            # self._vrise_count_dict that start with the cur_modName
            vRiseOrPower_list = [vRiseOrCapping_count_dict[cur_key]
                                 for cur_key in vRiseOrCapping_count_dict if cur_key.startswith(cur_mod_name)]
            try:  # In case vRiseOrPower_list is empty
                df_to_plot[cur_mod_name] = vRiseOrPower_list
            except ValueError:
                df_to_plot[cur_mod_name] = [np.nan] * len(df_to_plot)

        return df_to_plot

    def countplot(self,
                  count_dict_name,
                  fig_params=None):
        """ Plot the total number of :
            voltage rise above the defined threshold
            or the total number of capping commands
        for the simulation defined by the parameters of the last call of
        *.vRise_boxplot(*args)

        Parameters
        ----------
        count_dict_name : str
            'v_rise' : Plot of the total number of voltage rise above the defined
            threshold, 'capping': Plot the total number of capping command sent to
            the energy producer
        fig_params : (Figure axis, optional)
            Figure axis where to plot the current box plots

        """

        # _vrise_count_dict and _capping_count_dict are 2 dict created in
        #  _vRise_dict_as_dataFrame(*args)
        self._dict_vars = {'v_rise': '_vrise_count_dict',
                           'capping': '_capping_count_dict'
                           }  # Create a dict

        checker.check_countplot_inputs(self, count_dict_name)

        df_to_plot = self._count_dict_as_dataFrame(self._dict_vars[count_dict_name])

        if fig_params is None:
            x_len = len(df_to_plot) / 2  # Define len of figure depending on the number of
            # element in the df_to_plot
            fig_ = df_to_plot.plot(marker="*", ls='', figsize=(x_len, 4))
        else:
            fig_ = df_to_plot.plot(marker="*", ls='', ax=fig_params)
            fig_.legend('_')
        fig_.set_xticks(df_to_plot.index)
        fig_.set(ylabel='Count',
                 xlabel='HV Max Prod (MW)',
                 title=f'Bt+ = {self._bt} MW, {count_dict_name.capitalize()} ')
        fig_.grid(axis='x', lw=0.2)





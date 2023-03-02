.. 
     *TODO : load the ReadMe file in the tutorial file. For the moment inclusion is working fine, thanks to the extenxion m2r2, however the links in the md file 
     are broken in the rendered htlm page. Find a way to fix this problem. 
     Apparently the option `m2r_parse_relative_links` can help fix the problem however I don't know yet how to activate it in the conf.py file




.. _The voltage rise detection block scheme: https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Docs/VRiseControlBlockScheme.pdf



Tutorials
==========


.. warning:: 
     Please **READ** `The voltage rise detection block scheme_` before reading everything else.


This section presents a list of all the Notebooks or tutorials available. You may draw inspiration from them for the needed usability. This list is also available `here <https://github.com/pajjaecat/ORI-SRD/tree/main/Ressources/Notebooks#readme>`_



List of Tutorials
------------------


.. _CleanDataSTLaurentDeJourdes:

`CleanDataSTLaurentDeJourdes <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/CleanDataSTLaurentDeJourdes.ipynb>`_
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Clean the network's input data (Hv & Lv Prod, Load demand) and save the results to be recalled when needed.



.. _STLaurentDeJourdes_2:

`STLaurentDeJourdes_2 <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/STLaurentDeJourdes_2.ipynb>`_
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Simulations of the Voltage rise on Civaux's network when the controlled HV producer is:

* Not injecting into the network;
* Injecting into the network with no constraints considered;
* Injecting into the network with a maximum Voltage Rise threshold considered on the HV buses.


.. _2021_2022_KnownFuture:

`2021_2022_KnownFuture <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/2021_2022_KnownFuture.ipynb>`_
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Simulations using `The voltage rise detection block scheme_` with the block PRED having a perfect knowledge of the future in parallel to speed up the process.


.. _2021_2022_Persistence: 

`2021_2022_Persistence <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/2021_2022_Persistence.ipynb>`_ and `2021_2022_Persistence2 <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/2021_2022_Persistence2.ipynb>`_
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Simulations using `The voltage rise detection block scheme_` with the block **PRED**  using persistence of the previous instant as prediction method, respectively for :math:`max\_ vm\_ pu = 1.0250` and :math:`max\_vm\_pu = 1.0225`.

.. 
    .. _2021_2022_PersistenceRob:

    `2021_2022_PersistenceRob <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/2021_2022_PersistenceRob.ipynb>`_
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Simulations using the `The voltage rise detection block scheme <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Docs/VRiseControlBlockScheme.pdf>`_ with the block 
    **PRED** using persistence of the previous instant as prediction method.



    .. _RNN_StLaurentDeJourdes:

    `RNN_Train_StLaurentDeJourdes <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/RNN_Train_StLaurentDeJourdes.ipynb>`_ and `RNN_Sim_StLaurentDeJourdes <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/RNN_Sim_StLaurentDeJourdes.ipynb>`_
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    * Create and Train a Recurrent Neural Network `RNN <https://en.wikipedia.org/wiki/Recurrent_neural_network>`_ of type LSTM to predict some values; 
    * Use the previously trained RNN to predict the next :math:`\tilde{X}(k)` and :math:`\tilde{Y}(k)` based on the sliding history :math:`Z(k)` of the 
      past twelve daylight hours. `Figures <https://github.com/pajjaecat/ORI-SRD/tree/main/Ressources/Figures>`_ contains several comparison plots of the real variables 
      and their prediction.



    .. _2021_2022_RNN:

    `2021_2022_RNN <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/2021_2022_RNN.ipynb>`_
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Simulations using `The voltage rise detection block scheme_` with the block **PRED** based on the RNN created and trained in `RNN_StLaurentDeJourdes_` in parallel.




    .. _VoltageRiseBinaryUpdated:

    `VoltageRiseBinaryUpdated <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/VoltageRiseBinaryUpdated.ipynb>`_
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Using a power flow in parallel, compute the maximum voltage rise on the lower network (and convert it into a binary variable) from Jan-01-2020 to June-01-2022.


    .. _Numerical_VriseRNN:

    Numerical Voltage Rise RNN
    """""""""""""""""""""""""""
    * `VoltageRiseNum_Train <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/VoltageRiseNum_Train.ipynb>`_ - Create and Train an RNN to predict 
      the value of the maximum voltage rise.
    * `VoltageRiseNum_Pred <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/VoltageRiseNum_Pred.ipynb>`_ - Use the previously trained RNN to predict the
      maximum Voltage rise and compare the results to that of a simple power flow.



    .. _Binary_VriseRNN:

    Binary Voltage Rise RNN
    """""""""""""""""""""""""""
    * `VoltageRiseBin_Train <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/VoltageRiseBin_Train.ipynb>`_ - Create and train a RNN to Predict a binary variable
      (1 or 0), whether a voltage rise (above a predefined threshold i.e. 1.0250 ) will occur or not;
    * `VoltageRiseNum_Pred <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/VoltageRiseBin_Pred.ipynb>`_ - Use the previously trained RNN to predict whether a      voltage rise will occur or not in the next period.



    .. _2021_2022_RNN_Robust_All_Models: 

    `2021_2022_RNN_Robust_All_Models <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/2021_2022_RNN_Robust_All_Models.ipynb>`_
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Combine the prediction of three RNN models to evaluate whether an exceeding of the defined maximum voltage rise will occur in next step. The user with the help of paramUser get
    to choose the combination or the unique model to use. This is an implementation of the Combined RNN prediction described in Section 2.1 of `The voltage rise detection block scheme_` in parallel.



    .. _2021_2022_SimResAll_RNN:

    `2021_2022_SimResAll_RNN <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/2021_2022_SimResAll_RNN.ipynb>`_
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Compare the results of the simulations given by `2021_2022_RNN_Robust_All_Models_` when using each or a combination of the RNN models.



    .. _SensAnalysisP0100:

    `SensAnalysisP0100 <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/SensAnalysisP0100.ipynb>`_
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Perform a sensitivity analysis of the controlled Hv Prod maximum output depending on several prediction models:

    * Future Known;
    * Robust persistence model (of the previous period);
    * RNN Model1;
    * Robust RNN Model1;
    * Robust RNN Model 3. 

    We focus on the total Energy curtailed and the resulting voltage rise above a defined threshold.



    .. _SensAnalysisP0100_Res:

    `SensAnalysisP0100_Res <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/SensAnalysisP0100_Res.ipynb>`_
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    The results of the previously performed sensitivity analysis in `SensAnalysisP0100_` are extracted for analytics.






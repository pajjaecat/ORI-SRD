.. 
     *TODO : load the ReadMe file in the tutorial file. For the moment inclusion is working fine, thanks to the extenxion m2r2, however the links in the md file 
     are broken in the rendered htlm page. Find a way to fix this problem. 
     Apparently the option `m2r_parse_relative_links` can help fix the problem however I don't know yet how to activate it in the conf.py file

Tutorials
===========


.. warning:: 
     Please **READ** `The voltage rise detection block scheme <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Docs/VRiseControlBlockScheme.pdf>`_ before reading everything else.


This section presents a list of all the Notebooks or tutorials available. You may draw inspiration from them for the needed usability. This list is also available `here <https://github.com/pajjaecat/ORI-SRD/tree/main/Ressources/Notebooks#readme>`_




.. _CleanDataSTLaurentDeJourdes:

`CleanDataSTLaurentDeJourdes <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/CleanDataSTLaurentDeJourdes.ipynb>`_
**************************************************************************************************************************************
Clean the network's input data (Hv & Lv Prod, Load demand) and save the results to be recalled when needed.



.. _STLaurentDeJourdes_2:

`STLaurentDeJourdes_2 <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/STLaurentDeJourdes_2.ipynb>`_
*************************************************************************************************************************
Simulations of the Voltage rise on Civaux's network when the controlled HV producer is:

* Not injecting into the network;
* Injecting into the network with no constraints considered;
* Injecting into the network with a maximum Voltage Rise threshold considered on the HV buses.


.. _2021_2022_KnownFuture:

`2021_2022_KnownFuture <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/2021_2022_KnownFuture.ipynb>`_
**************************************************************************************************************************
Simulations using `The voltage rise detection block scheme <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Docs/VRiseControlBlockScheme.pdf>`_ with the block PRED having a perfect knowledge of the future in parallel to speed up the process.


.. 2021_2022_Persistence: 

`2021_2022_Persistence <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/2021_2022_Persistence.ipynb>`_ and `2021_2022_Persistence2 <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/2021_2022_Persistence2.ipynb>`_
****************************************************************************************************************************************************************************************************************************************************************************************
Simulations using `The voltage rise detection block scheme <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Docs/VRiseControlBlockScheme.pdf>`_ with the block **PRED**  using persistence of the previous instant as prediction method, respectively for :math:`max\\_ vm\\_ pu = 1.0250` and :math:`max\\_vm\\_pu = 1.0225`.


.. _2021_2022_PersistenceRob:

`2021_2022_PersistenceRob <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/2021_2022_PersistenceRob.ipynb>`_
**********************************************************************************
Simulations using the `The voltage rise detection block scheme <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Docs/VRiseControlBlockScheme.pdf>`_ with the block 
**PRED** using persistence of the previous instant as prediction method.



.. _RNN_StLaurentDeJourdes:

`RNN_Train_StLaurentDeJourdes <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/RNN_Train_StLaurentDeJourdes.ipynb>`_ and `RNN_Sim_StLaurentDeJourdes <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Notebooks/RNN_Sim_StLaurentDeJourdes.ipynb>`_
************************************************************************************************************************************************************************************************************************************************************************************************************************************

* Create and Train a Recurrent Neural Network `RNN <https://en.wikipedia.org/wiki/Recurrent_neural_network>`_ of type LSTM to predict some values; 
* Use the previously trained RNN to predict the next :math:`\tilde{X}(k)`and :math:`\tilde{Y}(k)` based on the sliding history :math:`Z(k)` of the 
  past twelve daylight hours. `Figures <https://github.com/pajjaecat/ORI-SRD/tree/main/Ressources/Figures>`_ contains several comparison plots of the real variables 
  and their prediction.



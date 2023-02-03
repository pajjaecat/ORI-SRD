All the files (data) called within the notebooks described here are located in [Pickle_files](../Pickle_files). When executing the notebooks, ensure the file location is set correctly (keep the same directory structure as this repository, and it should work just fine) and that all the compressed files (.zip, .7z) are decompressed in the corresponding folder. 
> As exemple decompress the content of the file [Pickle_files/ElectricNation.zip](../Pickle_files/ElectricNation.zip)  in a the same directory as  [Pickle_files/ElectricNation/]


</br> 


## Brief description of each  notebook

Make sure to read [Voltage Rise control Block scheme](../Docs/VRiseControlBlockScheme.pdf) to understand the notebooks well. 
***


##### [CleanDataSTLaurentDeJourdes](CleanDataSTLaurentDeJourdes.ipynb) 
> > Clean the network's input data (Hv & Lv Prod, Load demand) and save the results to be recalled when needed.

##### [STLaurentDeJourdes_2](STLaurentDeJourdes_2.ipynb)
> > Simulations of the Voltage rise on [Civaux's](../Pickle_files/CIVAUX.p) network when the controlled HV producer is: 
> > - not injecting into the network;
> > - injecting into the network with no constraints considered; 
> > - injecting into the network with a maximum Voltage Rise threshold considered on the HV buses.


##### [2021_2022_KnownFuture](2021_2022_KnownFuture.ipynb)
> > Simulations using the [Voltage Rise control Block scheme](../Docs/VRiseControlBlockScheme.pdf) with the block **PRED** having a perfect knowledge of the future in parallel to speed up the process.


##### [2021_2022_Persistence](2021_2022_Persistence.ipynb) and [2021_2022_Persistence2](2021_2022_Persistence2.ipynb)
> > Simulations using the [Voltage Rise control Block scheme](../Docs/VRiseControlBlockScheme.pdf) with the block **PRED** using persistence of the previous instant as prediction method, respectively for $max\\_ vm\\_ pu = 1.0250$ and $max\\_vm\\_pu = 1.0225$ 


##### [2021_2022_PersistenceRob](2021_2022_PersistenceRob.ipynb)
> > Simulations using the [Robust Voltage Rise control Block scheme](../Docs/VRiseControlBlockScheme.pdf) with the block **PRED** using persistence of the previous instant as prediction method.

##### [RNN_Train_StLaurentDeJourdes](RNN_Train_StLaurentDeJourdes.ipynb) and [RNN_Sim_StLaurentDeJourdes](RNN_Sim_StLaurentDeJourdes.ipynb)
> > - Create and Train a Recurrent Neural Network ([RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network)) of type LSTM to predict some values; 
> > - Use the previously trained RNN to predict the next $\tilde{X}(k)$ and $\tilde{Y}(k)$ based on the sliding history $Z(k)$ of the past twelve daylight hours. [Figures](../Figures) contains several comparison plots of the real variables and their prediction.

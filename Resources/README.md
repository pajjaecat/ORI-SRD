## Some definition

|    Definition             |                              Meaning                               
|   --------                |                ----------------------------------------            
| **LV\lv Prod\Sgens**      |   Lower voltage generators (producteurs BT)                        
| **HV\hv Prod or Sgens**   |   High voltage generators (producteurs HTA)
   **Upper Network**        |   The network where is located the Main Poste Source
   **Lower Network**        |   The network to study which is a small branch of the Upper Network


## Architecture
The available resource is subdivided into five subfolders as follows:
- [Excel_files](Excel_files/): Contains the the networks' inputs as ``.csv`` files 
  - Each of the Hv Prod must have its own associated files;
  - All of the Lv Prods (i.e. all the LV Prods in the Upper network) must be aggregated  in a unique file;
  - All of the Load  (i.e. all the load in the Upper network) must be aggregated in a unique file.
  
- [Figures](Figures/): Contains figures resulting from the simulations.

- [Pickle_files](Pickle_files/): Contains
  - The networks to simulate as ``.p`` files. Here we consider ST LAURENT and CIVAUX as respectively the upper and lower Network; 
  - [simulationResults](pickle_files/simulationResults), a folder where files/dataset associated with the notebooks or results of simulation that can be recall for easy use;
  
  - [RNN](Pickle_files/RNN) a folder where created [Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network) are saved.
  
- [Modules](Modules/): Contains different modules (classes, functions and variables) that are used in all the notebooks. See [py_filesReadMe](py_files/README.md) for more details

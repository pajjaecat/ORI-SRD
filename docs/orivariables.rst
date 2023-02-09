
#############################
Default variables
#############################
This section defines all the default variables used in the different notebooks. You may changed them globally in `oriVariables <https://github.com/pajjaecat/ORI-SRD/blob/1.0.2/Ressources/Modules/oriVariables.py>`_ depending on the problem specifities. 

.. automodule:: oriVariables


.. code:: python 

  # Create an attribute list to use in functions
  attr_list = [('bus', 'name'),
               ('load', 'bus'),
               ('switch', 'bus'),
               ('line', 'from_bus'),
               ('line', 'to_bus'),
               ('trafo', 'hv_bus'),
               ('trafo', 'lv_bus')]
     

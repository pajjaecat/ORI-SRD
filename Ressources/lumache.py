"""
Lumache - Python library for cooks and food lovers.

This is a Python docstring, we can use reStructuredText syntax here!

.. code-block:: python

    # Import lumache
    import lumache

    # Call its only function
    get_random_ingredients(kind=["cheeses"])
"""

__version__ = "0.1.0"


class InvalidKindError(Exception):
    """Raised if the kind is invalid."""

    pass


def get_random_ingredients(kind=None):
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]
    """
    return ["shells", "gorgonzola", "parsley"]



def readAndReshape_excelFile(f_name:str, 
                             folder_name:str, 
                             n_row2read:int=None):
    """
    Load both the lower (network used for opimization) and upper network, after which a configuration 
    of the main  parameters to use for the simulations are done. 
    Namely:
 
    Parameters 
    ----------
    upperNet_file:
        The upper Network file, with the approporiate extenxion (Must be present in the network_folder)
        Egg: 'ST LAURENT.p'
    lowerNet_file :
        The lower Network file, with the approporiate extenxion (Must be present in the network_folder)
        Egg:'CIVAUX.p'
    ctrld_hvProdName :
        Name of the controlled HV producer in the Lower Network. 
        Egg: 'P0100'
    params_coef_add_bt: 
        (0) coef_add_bt: float
            Value of the added output power for all the LV producers (MW) in the lower Network
        (1) coef_add_bt_dist: str
            How coef_add_bt is shared among the LV producers. 
            Three choices are possible
            (0): None (default) ==> No upscaling is done
            (1): 'uppNet' ==> coef_add_bt is added to the Sum of maximum output of all lower 
                 voltage (LV) producers (MW) in the upper Network. In consequence, the LV producers 
                 on the lower network receive only a fraction of coef_add_bt.
            (2): 'lowNet'==> coef_add_bt is added to the Sum of maximum output of all LV 
                 producers (MW) in the lower Network. In consequence, coef_add_bt is shared 
                 proportionnaly among all the LV producers on the lower network. 
            (3) 'lowNet_rand' ==> coef_add_bt is shared proportionnaly among a randomly selected 
                 set of the LV producers on the lower Network. The randomly selected set consist of 
                 half of all LV producers on the on the lower Network
    params_vRise:tuple 
        params_vRise[0]: tuple 
        Voltage Rise threshold associated with Higher voltages buses
            (0) vm_mu_max_hv: float
                Maximum authorised voltage rise of hv Buses on the Lower network
            (1) vm_mu_min_hv: float
                Minimum authorised voltage rise of hv Buses on the Lower network
        params_vRise[1]:tuple
        Voltage Rise threshold associated with lower voltages buses
            (0) vm_mu_max_lv: float
                Maximum authorised voltage rise of lv Buses on the Lower network
            (1) vm_mu_min_lv: float
                Minimum authorised voltage rise of lv Buses on the Lower network

    Return:
    -------
    
    
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

    return cols_to_read

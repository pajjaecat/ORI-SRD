""" Modules where all the checking function are defined """


def _check_countDictAsDf_input_inLocalSpace(self, input_var):
    """ Check if input_var that must be either of (1)self._vrise_count_dict or 
    (2)self._capping_count_dict is already present in the local space of the instance, 
    i.e. vRise_boxplot(*args) has already been executed once 
    """
    try:
        getattr(self, input_var)
    except AttributeError:
        raise Exception("""The method *.vRise_boxplot(*args) must be run before 
                        calling .*countplot(*args)""")
        
        
    
def _check_countplot_inputs(self, dict_name):
    """ Check if the input given by the user for countplot(*args) is authorised """    
    if dict_name not in self._dict_vars:
        raise Exception(f' The input must be either of {list(self._dict_vars.keys())} ')


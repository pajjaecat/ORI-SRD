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



# def readAndReshape_excelFile(f_name:str, 
#                              folder_name=excel_folder, 
#                              n_row2read:int=None):
#     """
#     Read and reshape in a one dimension array (that is returned) the excel file given by f_name
    
#     Parameters: 
#     ----------- 
#     f_name: str
#         Name of the file to load (with the correct extension)
#     folder_name: str
#         Location of the folder where the file is present
#     n_row2read : Int (default=0) 
#          Numbers of rows to read in the excel file.
#     """

#     filename = f"{folder_name}{f_name}"
#     cols_to_read = range(2, 8)  # Define index of columns to read 
#                                 # 0 10 20 30 40 50 
#                                 # Where each of the six columns to read represent a period.
#     input_data = pandas.read_csv(filename,
#                                  header=None,
#                                  sep=";",
#                                  usecols=cols_to_read,
#                                  nrows=n_row2read)

    return numpy.array(input_data).reshape(-1) / 1000  # /1000 To convert data (MW)

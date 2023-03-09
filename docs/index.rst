.. image:: ../Ressources/Figures/otherImages/SrdLogo.svg 
         :width: 185em
         :align: left
         

         

Welcome to SRD-ORI documentation!
=================================

Welcome to the official documentation of the `ODRI project <https://www.s2e2.fr/projets/odri/>`_  fifth axe **ORI** (Offre de Racordement Intelligente) or Flexible connexion proposal. 

-----------

------------

.. image:: https://readthedocs.org/projects/ori-srd/badge/?version=stable
    :target: https://ori-srd.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation Status
    
.. image:: https://img.shields.io/badge/python-3.7-blue.svg
    :target: https://www.python.org/downloads/release/python-370/
    
.. image:: ../Ressources/Figures/otherImages/version103.svg
    :target: https://ori-srd.readthedocs.io/en/1.0.3/
         
    
    
.. note::
   This project and page is under active development.

 
ORI's primary goal is to provide tools to help `SRD <https://www.srd-energies.fr/>`_ decide in which specificities an ORI should be proposed to new renewable energy producers demanding to be connected to the electric distribution Network. 

To help in this endeavour, we have developed the backbone of the project that is  `The voltage rise detection block scheme <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Docs/VRiseControlBlockScheme.pdf>`_.

.. warning:: 
   Please **READ** `The voltage rise detection block scheme <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Docs/VRiseControlBlockScheme.pdf>`_ before reading everything else.
   
To implement `The voltage rise detection block scheme <https://github.com/pajjaecat/ORI-SRD/blob/main/Ressources/Docs/VRiseControlBlockScheme.pdf>`_ ,
for new networks configuration please check out the sections : 
   - :doc:`howToUse` to get a step by step process to apply to any type of network;
   - :doc:`tutorials` to see the differents available tutorials to get inspired of for the implementation.
   
The remaining sections :doc:`orivariables`, :doc:`orifunctions` and :doc:`oriclasses` offer an in-depth view of the main functions used in the :doc:`tutorials`.  


.. 
         Check out the :doc:`usage` section for further information, including
         how to :ref:`installation` the project.





Contents
--------

.. toctree::
   :maxdepth: 2

   Home <self>
   HowToUse/howToUse
   Tutorials/tutorials
   orivariables/orivariables
   orifunctions/orifunctions
   oriclasses/oriclasses

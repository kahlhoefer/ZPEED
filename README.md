# ZPEED (Z' Exclusions from Experimental Data)

The ZPEED code calculates likelihoods and various test statistics for general Z' models.
 
Citation Policy
---------------

When using ZPEED, please cite
F. Kahlhoefer, A. Mück, S. Schulte and P. Tunney,
"Interference effects in dilepton resonance searches for Z' bosons and dark matter mediators",
2019, arXiv:1912.06374.

Running the code
----------------

ZPEED is written in python and works with both python2.7 and python3.5. It furthermore requires numpy and scipy. 

To run the code, simply do

  python RunMe.py 

Contact
-------

Any questions, comments, or suggestions should be directed to kahlhoefer@physik.rwth-aachen.de.

Contributors
------------

The following have contributed to the development and testing of ZPEED:

  * Felix Kahlhoefer, RWTH Aachen:
    Implementation of ATLAS likelihood and statistical method.

  * Stefan Schulte, MPP Munich:
    Implementation of cross section formulas, PDFs and detector effects.

  * Patrick Tunney, RWTH Aachen:
    Implementation of detector effects; testing and validation.

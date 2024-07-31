# THERMOELASTICITY

# Folder fem:   
The folder fem consists of the additional files and libraries that are required for performing finite element analysis in Elmer software.
The plasticity material property is incorporates in the FEM in addition to the elasticity property.

# Folder vae:   
The folder contains information about 

# STREAMLIT APP : specific enthalpy computation for given dataset of molar enthalpy 
The scope of the computation is currently limited to any compositions of binary, ternary or quaternary alloy composed of two or more of the Au, Ni, Ti, Zr elements
Thus, the  app is desiged on the basis of 4 element framework (Au, Nb, Ti, Zr)
The temperature and specific enthalpy (J/kg) is returned when a csv file consisting of T,H(J/mol) is uploaded. The headers T and H must not be included in the csv file while uploading.
The information about specific enthalpy is used in the finite element analysis.

[![Computation of Specific Enthalpy via Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://enthalpyautinbzr.streamlit.app/)

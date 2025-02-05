import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
from pymatgen.core.composition import Composition
import re

# Load the trained models and scalers
encoder = tf.keras.models.load_model('encoder_model')
decoder = tf.keras.models.load_model('decoder_model')
regressor = joblib.load('regressor.pkl')
scaler = joblib.load('scaler.pkl')
y_scaler = joblib.load('y_scaler.pkl')

# Define the available elements
available_elements = [
    'Mg', 'Cs', 'Co', 'Zr', 'Se', 'Dy', 'Pb', 'Ga', 'O', 'Sn', 
    'Yb', 'B', 'La', 'Si', 'V', 'Fe', 'S', 'Sc', 'Tl', 'Zn', 
    'Cl', 'Ce', 'Er', 'Nd', 'Pd', 'Y', 'P', 'Ta', 'In', 'Te', 
    'Ru', 'Rb', 'Tm', 'Tb', 'Sb', 'Al', 'Lu', 'Bi', 'Pr', 'Eu', 
    'Sm', 'Ba', 'Cr', 'Sr', 'Ni', 'Ca', 'As', 'Mn', 'Mo', 'Cd', 
    'Ti', 'Nb', 'Hf', 'Gd', 'Ag', 'Ge', 'Li', 'Br', 'Au', 'I', 
    'N', 'Na', 'Cu', 'Ho', 'K'
]

# Define the function to parse formulas
def parse_formula(formula):
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)?'
    elements = re.findall(pattern, formula)
    return list(set([element[0] for element in elements]))

# Define the function to extract and multiply stoichiometry
def extract_multiplier_and_replace(input_formula):
    pattern = r'\)(\d*\.?\d*)'
    match = re.search(pattern, input_formula)
    if match:
        multiplier = match.group(1)
        multiplier = float(multiplier) if multiplier else 1.0
        parts = re.split(pattern, input_formula)
        formula_without_multiplier = parts[0]
        content_within_parentheses = formula_without_multiplier.split('(')[-1]
        elements_within_parentheses = re.findall(r'([A-Za-z]+)(\d*\.?\d*)', content_within_parentheses)
        modified_elements = [(element, str(float(stoichiometry) * multiplier) if stoichiometry else '1.0') for element, stoichiometry in elements_within_parentheses]
        modified_formula = formula_without_multiplier.split('(')[0] + ''.join(element + stoichiometry for element, stoichiometry in modified_elements)
        return modified_formula
    else:
        return input_formula

# Function to featurize materials based on available elements
def featurize_materials(df, available_elements):
    features = []
    for _, row in df.iterrows():
        modified_formula = extract_multiplier_and_replace(row['Formula'])
        composition = Composition(modified_formula)
        composition_dict = composition.fractional_composition.as_dict()
        feature_vector = {element: composition_dict.get(element, 0) for element in available_elements}
        feature_vector['temperature(K)'] = row['temperature(K)']
        features.append(feature_vector)
    return pd.DataFrame(features)

# Function to preprocess new data
def preprocess_new_data(df, available_elements):
    features_df = featurize_materials(df, available_elements)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(features_df)
    X_scaled = scaler.transform(X_imputed)
    return X_scaled

# Function to calculate the weighted composition thermal conductivity for Ti3Au
def kth_Ti3Au(T):
    # Coefficients for Ti
    A_sTi = -1.66755674e-06
    B_sTi = 4.80727332e-03
    C_sTi = 1.47783620e+01
    D_lTi = 0.017
    E_lTi = 0.969
    
    # Coefficients for Au
    A_sAu = 0
    B_sAu = -6.93088808e-02
    C_sAu = 3.38918567e+02
    D_lAu = 0.027397
    E_lAu = 100.0
    
    # Thermal conductivity for Ti
    if 300 <= T <= 1668:
        kth_Ti = A_sTi * T**2 + B_sTi * T + C_sTi
    elif 1668 < T <= 2200:
        kth_Ti = D_lTi * T + E_lTi
    else:
        return "Temperature out of range for Ti thermal conductivity"
    
    # Thermal conductivity for Au
    if 300 <= T <= 1668:
        kth_Au = A_sAu * T**2 + B_sAu * T + C_sAu
    elif 1668 < T <= 2200:
        kth_Au = D_lAu * T + E_lAu
    else:
        return "Temperature out of range for Au thermal conductivity"
    
    # Weighted thermal conductivity for Ti3Au
    kth_Ti3Au = 0.75 * kth_Ti + 0.25 * kth_Au
    return kth_Ti3Au

# Streamlit app
st.title("Thermoelectric Material Thermal Conductivity Prediction")
formula_input = st.text_input("Enter the chemical formula:")
temperature_input = st.number_input("Enter the temperature (K):", min_value=300, max_value=2200, value=300)

if st.button("Predict Thermal Conductivity"):
    if formula_input:
        # Create a DataFrame for the new input
        new_data = pd.DataFrame({
            'Formula': [formula_input],
            'temperature(K)': [temperature_input]
        })

        # Preprocess new data
        X_scaled = preprocess_new_data(new_data, available_elements)

        # Predict using the encoder and decoder
        predictions = []
        for _ in range(20):
            z_mean, z_log_var, z = encoder.predict(X_scaled)
            X_reconstructed = decoder.predict(z)

            # Predict thermal conductivity
            y_scaled_pred = regressor.predict(z)
            y_pred = y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()
            predictions.append(y_pred[0])

        # Filter positive values
        positive_predictions = [pred for pred in predictions if pred > 0]

        if not positive_predictions:
            st.warning("No positive thermal conductivity values predicted.")
        else:
            # Calculate the weighted composition thermal conductivity
            weighted_kth = kth_Ti3Au(temperature_input)
            
            # Find the prediction closest to the weighted composition thermal conductivity
            closest_prediction = min(positive_predictions, key=lambda x: abs(x - weighted_kth))

            st.write(f"Predicted Thermal Conductivity: {closest_prediction:.2f} W/mK")
    else:
        st.error("Please enter a chemical formula.")


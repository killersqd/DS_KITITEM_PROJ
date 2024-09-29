import streamlit as st
import pandas as pd

# Load your dataset
df = pd.read_csv(r"G:\class is started\PROJECT\final arrangement\last_&_best_model_predictions.csv")

# Sidebar with dropdown for kit items and forecasted month slider
selected_kit_item = st.sidebar.selectbox('Select Kit Item', df['Kit Item'].unique())
forecasted_month = st.sidebar.slider('Select Forecasted Month', 1, 2, 3)

# Filter the dataset based on the selected kit item
selected_item_data = df[df['Kit Item'] == selected_kit_item]

# Display the selected kit item and its best model and MAPE
st.write(f"Selected Kit Item: {selected_kit_item}")
st.write(f"Best Model: {selected_item_data['Best_Model'].values[0]}")
st.write(f"Best MAPE: {selected_item_data['Best_MAPE'].values[0]}")

# Display the prediction for the selected forecasted month
predicted_values = selected_item_data.iloc[0]['Predicted_Values']
predicted_value = predicted_values[forecasted_month - 1]
st.write(f"Predicted Value for Month {forecasted_month}: {predicted_value}")



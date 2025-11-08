import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load Model Assets ---
@st.cache_resource
def load_assets():
    with open('student_pass_fail_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_assets()

# --- 2. Prediction Function ---
def predict_student_status(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame of student marks, scales them, and predicts PASS/FAIL.
    Returns the input DataFrame with an added 'Predicted Status' column.
    """
    if input_data.empty:
        return pd.DataFrame()

    # Ensure the input DataFrame has the same columns and order as the training data
    # Handle June Exam % NaN values similar to training
    input_data_processed = input_data.copy()
    if 'June Exam %' in input_data_processed.columns:
        # Use median from training data if available, or just the input data's median
        # For a robust solution, you'd save the training median in a pickle too.
        # For simplicity here, we'll use the input's median for new data if it has NaNs.
        # A better approach for production would be to use the median from the *training* data.
        if input_data_processed['June Exam %'].isnull().any():
             st.warning("Missing 'June Exam %' values detected. Imputing with median of provided data.")
             input_data_processed['June Exam %'].fillna(input_data_processed['June Exam %'].median(), inplace=True)
    else:
        st.error("Missing 'June Exam %' column in the uploaded file. Please ensure all required columns are present.")
        return pd.DataFrame() # Return empty if critical column is missing

    # Reorder columns to match the trained features
    try:
        input_data_ordered = input_data_processed[feature_names]
    except KeyError as e:
        st.error(f"Missing required feature column in the uploaded file: {e}. Expected: {', '.join(feature_names)}")
        return pd.DataFrame()

    # Scale the input data
    scaled_data = scaler.transform(input_data_ordered)

    # Make predictions
    predictions = model.predict(scaled_data)

    # Add predictions to the original input DataFrame
    input_data['Predicted Status'] = predictions
    return input_data

# --- 3. Streamlit UI ---
st.set_page_config(page_title="Student Pass/Fail Prediction", layout="centered")

st.title("👨‍🎓 Student Pass/Fail Prediction App")
st.markdown("""
This application predicts whether a student will PASS or FAIL based on their academic marks.
You can either enter marks manually or upload an Excel file for bulk prediction.
""")

st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Choose your preferred input method:",
    ("Manual Entry", "Upload Excel File")
)

# Display expected features
st.sidebar.subheader("Expected Features")
st.sidebar.info("Please ensure your input contains the following columns (case-sensitive) and in the correct order for manual entry:")
for i, feature in enumerate(feature_names, 1):
    st.sidebar.write(f"  {i}. **{feature}**")

# --- Manual Entry Section ---
if input_method == "Manual Entry":
    st.header("Manual Mark Entry")
    st.write("Enter the student's marks below:")

    # Create input fields for each feature
    input_values = {}
    col1, col2 = st.columns(2)
    for i, feature in enumerate(feature_names):
        if i % 2 == 0:
            with col1:
                input_values[feature] = st.number_input(f"{feature} (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1, key=f"manual_{feature}")
        else:
            with col2:
                input_values[feature] = st.number_input(f"{feature} (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1, key=f"manual_{feature}")

    if st.button("Predict Status (Manual Entry)"):
        # Create a DataFrame from manual inputs
        manual_df = pd.DataFrame([input_values])

        # Make prediction
        result_df = predict_student_status(manual_df)

        if not result_df.empty:
            predicted_status = result_df['Predicted Status'].iloc[0]
            st.subheader("Prediction Result:")
            if predicted_status == 'PASS':
                st.success(f"The student is predicted to **{predicted_status}**! 🎉")
            else:
                st.error(f"The student is predicted to **{predicted_status}**! 😞")
            st.dataframe(manual_df.drop('Predicted Status', axis=1), use_container_width=True)


# --- Upload Excel File Section ---
elif input_method == "Upload Excel File":
    st.header("Upload Excel File for Bulk Prediction")
    st.write("Upload an Excel file (.xlsx) containing student marks. Ensure columns match the expected features.")

    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        try:
            bulk_df = pd.read_excel(uploaded_file)
            st.subheader("Preview of Uploaded Data:")
            st.dataframe(bulk_df.head(), use_container_width=True)

            if st.button("Predict Status (Bulk Upload)"):
                with st.spinner("Making predictions..."):
                    # Make predictions
                    predicted_df = predict_student_status(bulk_df.copy()) # Use a copy to avoid modifying original

                if not predicted_df.empty:
                    st.subheader("Prediction Results:")
                    st.dataframe(predicted_df, use_container_width=True)

                    # Option to download results
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv = convert_df_to_csv(predicted_df)

                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='student_predictions.csv',
                        mime='text/csv',
                    )
                else:
                    st.error("Could not generate predictions. Please check your file and the console for errors.")

        except Exception as e:
            st.error(f"Error reading the Excel file or making predictions: {e}")
            st.info("Please ensure your Excel file is correctly formatted and contains the expected columns.")
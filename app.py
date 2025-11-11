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
    
    # Store a robust default/global median for June Exam % from training data.
    # In a real scenario, this would be computed during model training and saved
    # (e.g., pickle `X_train['June Exam %'].median()` and load it here).
    # For demonstration, using a sensible default.
    global_june_exam_median = 50.0 
    
    return model, scaler, feature_names, global_june_exam_median

model, scaler, feature_names, global_june_exam_median = load_assets()

# --- 2. Data Validation Function ---
def validate_input_data(df_input: pd.DataFrame, is_manual_entry: bool = False) -> (bool, pd.DataFrame, str):
    """
    Validates the input DataFrame for correct columns, data types, and value ranges.
    Returns (True, cleaned_df, "") on success, or (False, None, error_message) on failure.
    """
    cleaned_df = df_input.copy()
    errors = []

    # 1. Check for required columns
    missing_columns = [col for col in feature_names if col not in cleaned_df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}. Please ensure your input contains all expected features.")
        return False, None, "\n".join(errors)

    # 2. Ensure columns are in the correct order for prediction
    try:
        cleaned_df = cleaned_df[feature_names]
    except KeyError as e:
        errors.append(f"Column reordering failed: {e}. Ensure all feature names are exact matches.")
        return False, None, "\n".join(errors)

    # 3. Data type and range validation
    for col in feature_names:
        # Try to convert to numeric, coercing errors
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

        # Check for non-numeric values after coercion.
        # If NaNs are introduced, and it's NOT 'June Exam %', we fail immediately.
        # 'June Exam %' NaNs are handled specifically later.
        if cleaned_df[col].isnull().any() and col != 'June Exam %':
            invalid_rows = cleaned_df[cleaned_df[col].isnull()].index.tolist()
            if is_manual_entry:
                errors.append(f"'{col}' must be a numeric value.")
            else:
                errors.append(f"Non-numeric values found in column '{col}' at row(s): {', '.join([str(r + 2) for r in invalid_rows])} (Excel row numbers).")
            return False, None, "\n".join(errors)

        # Check for out-of-range values (assuming marks are percentages 0-100)
        # This check should only apply to non-NaN values.
        if not cleaned_df[col].isnull().all() and not ((cleaned_df[col] >= 0) & (cleaned_df[col] <= 100)).all():
            # Filter for rows that are actually out of range AND not null
            invalid_rows = cleaned_df[~((cleaned_df[col] >= 0) & (cleaned_df[col] <= 100)) & cleaned_df[col].notnull()].index.tolist()
            if invalid_rows: # Only add error if there are actual out-of-range, non-null values
                if is_manual_entry:
                    errors.append(f"'{col}' must be between 0 and 100.")
                else:
                    errors.append(f"Values in column '{col}' are outside the 0-100 range at row(s): {', '.join([str(r + 2) for r in invalid_rows])} (Excel row numbers).")
                return False, None, "\n".join(errors)

    # 4. Missing value handling for 'June Exam %'
    if 'June Exam %' in cleaned_df.columns and cleaned_df['June Exam %'].isnull().any():
        original_null_rows = cleaned_df[cleaned_df['June Exam %'].isnull()].index.tolist()
        imputation_value = None

        if is_manual_entry:
            # For manual entry, st.number_input usually defaults to a value (e.g., 50.0).
            # If a user explicitly clears it, it might become NaN. We'll impute with a fixed value.
            imputation_value = 50.0
            st.warning("'June Exam %' was empty, imputed with 50.0 for manual entry.")
        else: # Bulk upload path
            # Calculate median from the *current batch's non-null* 'June Exam %' values
            batch_median = cleaned_df['June Exam %'].median()
            
            if pd.isna(batch_median):
                # If all 'June Exam %' values in the uploaded batch are NaN, use the global fallback median.
                imputation_value = global_june_exam_median
                st.warning(f"All 'June Exam %' values in the uploaded file were missing. Imputing with global fallback median ({imputation_value:.2f}).")
            else:
                # Use the median of the current batch
                imputation_value = batch_median
                st.warning(f"Missing 'June Exam %' values detected at row(s): {', '.join([str(r + 2) for r in original_null_rows])}. Imputing with median ({imputation_value:.2f}) of provided data.")
        
        # Perform the imputation
        cleaned_df['June Exam %'].fillna(imputation_value, inplace=True)

    # Final check for any remaining missing values in any column.
    # This is critical to ensure no NaNs are passed to the model.
    if cleaned_df.isnull().any().any():
        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().any():
                invalid_rows = cleaned_df[cleaned_df[col].isnull()].index.tolist()
                errors.append(f"Missing values still found in column '{col}' at row(s): {', '.join([str(r + 2) for r in invalid_rows])}. Please ensure all required fields are filled or correctly imputed.")
        return False, None, "\n".join(errors)

    if errors: # This block should ideally not be reached if previous returns work correctly.
        return False, None, "\n".join(errors)

    return True, cleaned_df, ""


# --- 3. Prediction Function ---
def predict_student_status(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame of student marks, scales them, and predicts PASS/FAIL.
    Returns the input DataFrame with an added 'Predicted Status' column.
    Assumes data is already validated and cleaned.
    """
    if input_data.empty:
        return pd.DataFrame()

    # Scale the input data
    scaled_data = scaler.transform(input_data)

    # Make predictions
    predictions = model.predict(scaled_data)

    # Add predictions to the original input DataFrame
    input_data_with_predictions = input_data.copy()
    input_data_with_predictions['Predicted Status'] = predictions
    return input_data_with_predictions

# --- 4. Streamlit UI ---
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
st.sidebar.subheader("Expected Features (0-100%)")
st.sidebar.info("Please ensure your input contains the following columns with numeric values between 0 and 100 (case-sensitive):")
for i, feature in enumerate(feature_names, 1):
    st.sidebar.write(f"  {i}. **{feature}**")

# --- Manual Entry Section ---
if input_method == "Manual Entry":
    st.header("Manual Mark Entry")
    st.write("Enter the student's marks below:")

    # Create input fields for each feature, using two columns for better layout
    cols = st.columns(2)
    col_idx = 0

    for i, feature in enumerate(feature_names):
        default_value = 50.0 # Sensible default
        with cols[col_idx]:
            input_values[feature] = st.number_input(f"{feature} (%)", min_value=0.0, max_value=100.0, value=default_value, step=0.1, key=f"manual_{feature}")
        col_idx = 1 - col_idx # Switch column for next input

    if st.button("Predict Status (Manual Entry)"):
        # Create a DataFrame from manual inputs
        manual_df = pd.DataFrame([input_values])

        # --- Data Validation for Manual Entry ---
        is_valid, validated_df, error_message = validate_input_data(manual_df, is_manual_entry=True)

        if is_valid:
            # Make prediction
            result_df = predict_student_status(validated_df)

            if not result_df.empty:
                predicted_status = result_df['Predicted Status'].iloc[0]
                st.subheader("Prediction Result:")
                if predicted_status == 'PASS':
                    st.success(f"The student is predicted to **{predicted_status}**! 🎉")
                else:
                    st.error(f"The student is predicted to **{predicted_status}**! 😞")
                # Display all input features and the prediction
                st.dataframe(result_df, use_container_width=True) 
            else:
                st.error("Prediction function returned an empty DataFrame.")
        else:
            st.error(f"Validation Error for Manual Entry:\n{error_message}")


# --- Upload Excel File Section ---
elif input_method == "Upload Excel File":
    st.header("Upload Excel File for Bulk Prediction")
    st.write("Upload an Excel file (.xlsx) containing student marks. Ensure columns match the expected features and values are numeric percentages between 0-100.")

    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        try:
            bulk_df = pd.read_excel(uploaded_file)
            st.subheader("Preview of Uploaded Data:")
            st.dataframe(bulk_df.head(), use_container_width=True)

            if st.button("Predict Status (Bulk Upload)"):
                with st.spinner("Validating and making predictions..."):
                    # --- Data Validation for Bulk Upload ---
                    is_valid, validated_df, error_message = validate_input_data(bulk_df.copy(), is_manual_entry=False)

                    if is_valid:
                        # Make predictions
                        predicted_df = predict_student_status(validated_df)

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
                            st.error("Could not generate predictions. The prediction function returned an empty DataFrame.")
                    else:
                        st.error(f"Validation Error for Uploaded File:\n{error_message}")

        except Exception as e:
            st.error(f"An unexpected error occurred while reading the Excel file or during prediction: {e}")
            st.info("Please ensure your Excel file is correctly formatted (e.g., no merged cells, valid sheet name if specified).")

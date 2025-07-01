import requests
import pandas as pd

# Define your sample input with all required fields
sample_input = {
    "Income": 50000.0,
    "Education level": 3.0,  # Ordinal encoded value (e.g., Higher education)
    "Age": 35.0,
    "Employment length": 5.0,
    "Family member count": 3.0,
    "Gender_Female": 1,
    "Gender_Male": 0,
    "Marital status_Civil marriage": 0,
    "Marital status_Married": 1,
    "Marital status_Separated": 0,
    "Marital status_Single / not married": 0,
    "Marital status_Widow": 0,
    "Dwelling_Co-op apartment": 0,
    "Dwelling_House / apartment": 1,
    "Dwelling_Municipal apartment": 0,
    "Dwelling_Office apartment": 0,
    "Dwelling_Rented apartment": 0,
    "Dwelling_With parents": 0,
    "Employment status_Commercial associate": 0,
    "Employment status_Pensioner": 0,
    "Employment status_State servant": 1,
    "Employment status_Student": 0,
    "Employment status_Working": 0,
    "Has a car_No": 0,
    "Has a car_Yes": 1,
    "Has a property_No": 0,
    "Has a property_Yes": 1,
    "Has a work phone_No": 0,
    "Has a work phone_Yes": 1,
    "Has a phone_No": 1,
    "Has a phone_Yes": 0,
    "Has an email_No": 0,
    "Has an email_Yes": 1
}

# Convert to DataFrame with exactly one row
input_df = pd.DataFrame([sample_input])

# Send to MLflow model server
response = requests.post(
    url="http://127.0.0.1:5005/invocations",
    headers={"Content-Type": "application/json"},
    json={"dataframe_records": input_df.to_dict(orient="records")}
)

# Print result
if response.status_code == 200:
    print("✅ Prediction:", response.json())
else:
    print("🚫 Failed to get prediction")
    print("Status Code:", response.status_code)
    print("Response:", response.text)


# Employment length input slider
st.write("""
## Employment length
""")
input_employment_length = np.negative(
    st.slider(
        "Select your employment length", value=6, min_value=0, max_value=30, step=1
    )
    * 365.25
)

# Income
st.write("""
## Income
""")
input_income = int(st.text_input("Enter your income (in USD)", 0))
income = st.number_input("Income (USD)", min_value=0, value=50000)


# Family member count
st.write("""
## Family member count
""")
fam_member_count = float(
    st.selectbox("Select your family member count", [1, 2, 3, 4, 5, 6])
)
family_members = st.number_input("Family Member Count", min_value=1, value=2)

# Age input slider
st.write("""
## Age
""")
input_age = np.negative(
    st.slider("Select your age", value=42, min_value=18, max_value=70, step=1) * 365.25
)

age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
emp_length = st.number_input("Employment Length (years)", min_value=0.0, step=0.1, value=2.0)
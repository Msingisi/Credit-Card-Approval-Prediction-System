import pandas as pd
import os

def process_and_clean_credit_data(app_path: str, credit_path: str, output_dir: str = "clean_data") -> None:
    application_df = pd.read_csv(app_path)
    credit_df = pd.read_csv(credit_path)
    begin_month = credit_df.groupby('ID', as_index=False)['MONTHS_BALANCE'].agg('min')
    begin_month = begin_month.rename(columns={'MONTHS_BALANCE': 'Account age'})
    application_df = pd.merge(application_df, begin_month, how='left', on='ID')
    credit_df['dep_value'] = None
    credit_df.loc[credit_df['STATUS'].isin(['2', '3', '4', '5']), 'dep_value'] = 'Yes'
    cpunt = credit_df.groupby('ID')['dep_value'].count().reset_index()
    cpunt['dep_value'] = cpunt['dep_value'].apply(lambda x: 'Yes' if x > 0 else 'No')
    application_df = pd.merge(application_df, cpunt, how='inner', on='ID')
    application_df['Is high risk'] = application_df['dep_value'].map({'Yes': 1, 'No': 0})
    application_df.drop('dep_value', axis=1, inplace=True)

    application_df = application_df.rename(columns={
        'CODE_GENDER': 'Gender',
        'FLAG_OWN_CAR': 'Has a car',
        'FLAG_OWN_REALTY': 'Has a property',
        'CNT_CHILDREN': 'Children count',
        'AMT_INCOME_TOTAL': 'Income',
        'NAME_INCOME_TYPE': 'Employment status',
        'NAME_EDUCATION_TYPE': 'Education level',
        'NAME_FAMILY_STATUS': 'Marital status',
        'NAME_HOUSING_TYPE': 'Dwelling',
        'DAYS_BIRTH': 'Age',
        'DAYS_EMPLOYED': 'Employment length',
        'FLAG_MOBIL': 'Has a mobile phone',
        'FLAG_WORK_PHONE': 'Has a work phone',
        'FLAG_PHONE': 'Has a phone',
        'FLAG_EMAIL': 'Has an email',
        'OCCUPATION_TYPE': 'Job title',
        'CNT_FAM_MEMBERS': 'Family member count',
        'Account age': 'Account age'
    })

    columns_to_rename = [
        "Gender", "Has a car", "Has a property", 
        "Has a mobile phone", "Has a work phone", 
        "Has a phone", "Has an email"
    ]
    application_df[columns_to_rename] = application_df[columns_to_rename].replace({
        "M": "Male", "F": "Female", 
        "Y": "Yes", "N": "No", 
        1: "Yes", 0: "No"
    })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cleaned_credit_data.csv")
    application_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    process_and_clean_credit_data(
        app_path="data/application_record.csv",
        credit_path="data/credit_record.csv"
    )
# Import packages
import pandas as pd
import streamlit as st
import os
import zipfile
import shutil

# Load loan-level data
@st.cache_resource
def load_loan_data(file_path):
    extract_dir = 'zip'
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    pkl_file_name = os.path.basename(file_path).replace('.zip', '.pkl')
    pkl_file_path = os.path.join(extract_dir, pkl_file_name)
    df = pd.read_pickle(pkl_file_path)
    os.remove(pkl_file_path)
    shutil.rmtree(extract_dir)
    df['Date'] = pd.to_datetime(df['Date'])
    df['ApprovalDate'] = pd.to_datetime(df['ApprovalDate'])
    df_saved = df.copy()
    return df, df_saved

# Load loan-level data
@st.cache_resource
def load_cpr_model(file_path):
    model = pd.read_pickle(file_path)
    return model

# Load dictionary
@st.cache_resource
def load_dictionary(file_path):
    dictionary = pd.read_excel(file_path)
    new_begin_rows = pd.DataFrame({
        'Field Name': ['LoanID', 'Date'],
        'Definition': ['Unique identifier for each loan', 
                       'Observation month'],
    })
    new_end_rows = pd.DataFrame({
        'Field Name': ['MaturityDate', 'Prepayment', 'ChargeOff', 'Loan Age', 'Obs Market Rate', 'Orig Market Rate',
                       'Incentive', 'Model Prepayment', 'UnempRate', 'US10YrTRate'],
        'Definition': ['Maturity date interpretted from ApprovalDate and TermInMonths',
                       '1 if prepaid on this record, 0 otherwise',
                       '1 if charged off on this record, 0 otherwise',
                       'Months from ApprovalDate to observation date',
                       'Average SBA 504 25 Yr Term new origination interest rate on observation date',
                       'Average SBA 504 25 Yr Term new origination interest rate on ApprovalDate',
                       'Orig Market Rate - Obs Market Rate',
                       'Monthly probability of prepayment from xgboost model using Loan Age, Incentive, GrossApproval and UnempRate',
                       'US national unemployment rate on Date',
                       'US 10 Year Treasury yield on Date'],
    })
    dictionary = pd.concat([new_begin_rows, dictionary, new_end_rows]).reset_index(drop=True)
    columns_to_keep = ['Date', 'LoanID', 'ThirdPartyDollars', 'GrossApproval', 'ApprovalDate', 'DeliveryMethod', 'subpgmdesc', 'TermInMonths',
                   'NaicsDescription', 'ProjectState', 'BusinessType', 'BusinessAge', 'JobsSupported', 'MaturityDate', 'Prepayment', 'ChargeOff',
                   'Loan Age', 'Obs Market Rate', 'Orig Market Rate', 'Incentive', 'Model Prepayment', 'UnempRate', 'US10YrTRate']
    return dictionary[dictionary['Field Name'].isin(columns_to_keep)]

# Submit query to gpt
def make_api_call(client, primer, user_prompt):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

# Function to concatenate previous interactions with the new prompt
def build_prompt(previous_interactions, new_user_input):
    return previous_interactions + "\n" + new_user_input

# Function to move results explanation to the end
def move_explanation(response):

    # Split the response into lines
    lines = response.split('\n')

    # Find the index where "Results Explanation" is mentioned
    start_index = -1
    end_index = -1
    in_block = False

    for i, line in enumerate(lines):
        if line.strip() == 'with st.expander("Results Explanation"):':
            start_index = i
            in_block = True
        elif in_block:
            if line.strip().endswith('`)') or line.strip().endswith('")'):
                end_index = i
                in_block = False

    # Move the "Results Explanation" code block to the end
    if start_index != -1 and end_index != -1:
        expander_code = lines[start_index:end_index + 1]
        del lines[start_index:end_index + 1]
        lines.extend(expander_code)

    # Reassemble the modified script
    response = '\n'.join(lines)

    # Return new response
    return response
# Import packages
import streamlit as st
import os
from openai import OpenAI
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import seaborn as sns
import zipfile
import shutil
import xgboost as xgb
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 800)

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

# Load dictionary
@st.cache_resource
def load_dictionary(file_path):
    dictionary = pd.read_excel(file_path)
    new_rows = pd.DataFrame({
        'Field Name': ['LoanID', 'Date', 'MaturityDate', 'Prepayment', 'ChargeOff', 'Loan Age', 'Obs Market Rate', 'Orig Market Rate',
                       'Incentive', 'Model Prepayment'],
        'Definition': ['Unique identifier for each loan', 
                       'Observation month',
                       'Maturity date interpretted from ApprovalDate and TermInMonths',
                       '1 if prepaid on this record, 0 otherwise',
                       '1 if charged off on this record, 0 otherwise',
                       'Months from ApprovalDate to observation date',
                       'Average SBA 504 25 Yr Term new origination interest rate on observation date',
                       'Average SBA 504 25 Yr Term new origination interest rate on ApprovalDate',
                       'Orig Market Rate - Obs Market Rate',
                       'Monthly probability of prepayment from xgboost model using Loan Age, Incentive, and GrossApproval'],
    })
    dictionary = pd.concat([new_rows, dictionary]).reset_index(drop=True)
    columns_to_keep = ['Date', 'LoanID', 'ThirdPartyDollars', 'GrossApproval', 'ApprovalDate', 'DeliveryMethod', 'subpgmdesc', 'TermInMonths',
                   'NaicsDescription', 'ProjectState', 'BusinessType', 'BusinessAge', 'JobsSupported', 'MaturityDate', 'Prepayment', 'ChargeOff',
                   'Loan Age', 'Obs Market Rate', 'Orig Market Rate', 'Incentive', 'Model Prepayment']
    return dictionary[dictionary['Field Name'].isin(columns_to_keep)]

# Set up gpt
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key = openai_api_key)

# Set primer to prepare gpt for queries
primer = """You are a helpful assistant. 
            I will ask you for python scripts. 
            These scripts will deal with a dataframe called df. Do not edit the original df dataframe.
            This dataframe has columns Date, LoanID, Prepayment, ChargeOff, and Loan Age, amongst other columns.
            There is a non-numeric column BusinessAge in the dataframe.
            Prepayment and ChargeOff are either 1 or 0.
            Only return the python script, do not return any text explanations.
            The python script should first use "with st.expander("Results Explanation"):" and st.write() to write a summary explanation of what you understand the question to be, and a concise description of how the python code works.
            Do not return any imports, assume that I have already imported all necessary packages.
            CPR is calculated as CPR = 1 - (1 - HP) ^ 12, where HP equals the average of the Prepayment column.
            CDR is calculated as CDR = 1 - (1 - HC) ^ 12, where HC equals the average of the ChargeOff column.
            When displaying CDR or CPR in a plot or table, format the CDR or CPR as a percentage to two decimal points using ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2)).
            After using groupby, check if the groupby variable has "Date" in the name. If it does, transform this variable in the groupby result using ".dt.date".
            There is a column called Model Prepayment that contains a model probability that Prepayment is 1.
            Model CPR can be calculated by calculating HP using the average of the Model Prepayment column instead.
            Do not use plt.yticks().
            There is a streamlit that already exists, all results will be printed to this streamlit.
            Do not use df.groupby().mean().
            Only run mean() on specific columns, because some columns in df are non-numeric.
            When using groupby(), use a list to refer to multiple columns, do not use a tuple. For example, use `df.groupby('Column1')[['Column2', 'Column3']].mean()`.
            To round a column to the nearest VALUE, use something like .apply(lambda x: np.round(x / VALUE) * VALUE).
            Refer to matplotlib.ticker as mtick if you use it.
            Do not call st.pyplot without an argument, this will be deprecated soon.
            If you are asked to plot, create a line plot without markers, make sure it includes a title and axis names, and show the plot on the streamlit using st.pyplot.
            If you plot, make sure the x-axis labels are rotated if they are long, and use ha="right".
            If you plot and use strings for the x-axis labels, make sure there are not too many labels so they don't overlap.
            If you plot actuals, plot them in different shades of blue. If you plot model, plot them in different shades of red.
            Make sure all lines are a different color.
            If you need to calculate the difference between two dates in months, do this directly using dt.year and dt.month.
            If the user asks what you are able to do, write to the streamlit that you are able to transform natural language queries into python code that can be used to query a dataframe
            of SBA 504 historical data, and potentially create plots and other graphics.
            If you want to write a message, make sure to write code that writes the message to the streamlit.
            If and only if you are asked to plot bars on a rounded x-axis variable, adjust the bar width to be 80% of the rounding interval.
            Do not plot bars unless you are asked to. By default all plots should be line plots unless otherwise requested.
            Do not train any machine learning models like xgboost, logistic regression, etc. under any circumstances.
            If you are asked to train a machine learning model, do not do it, instead print to streamlit that you are not allowed to do this."""

# Additional primer to be ended at the end of the prompt
prompt_addition = """"""

# Create streamlit app and take in queries
def main():

    # Set streamlit title
    st.title("SBA 504 Data Analysis with GPT")

    # Load data only once, using the cached function
    df, df_saved = load_loan_data('sbadata_dyn_small.zip')
    dictionary = load_dictionary('7a_504_foia-data-dictionary.xlsx')

    # Sidebar for navigation using radio buttons
    page = st.sidebar.radio("Menu", ["Chat", "User Guide", "Dictionary"])

    # Choose page
    if page == "Chat":
        display_chat(df, df_saved)  
    elif page == "User Guide":
        display_user_guide()
    elif page == "Dictionary":
        display_dictionary(dictionary)  

# Create chat page
def display_chat(df, df_saved):

    # Global variables to pass to exec
    exec_globals = {'df': df, 'pd': pd, 'plt': plt, 'mtick': mtick, 'mpl': mpl, 'st': st, 'np': np, 'xgb': xgb,
                    'MaxNLocator': MaxNLocator, 'mdates': mdates, 'sns': sns, 'train_test_split': train_test_split}

    # Initialize 'previous_interactions' in session_state if not present
    if 'previous_interactions' not in st.session_state:
        st.session_state['previous_interactions'] = ""

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user" or message["content"] == "Please try a different query.":
                st.markdown(message["content"])
            else:
                exec(message["content"], exec_globals)
                with st.expander("Python Script"):
                    st.code(message["content"], language='python')

    # Accept user input
    if prompt := st.chat_input("Type your prompt here..."):

        # Remove white space from prompt
        prompt = prompt.strip()

        # Make sure prompt ends with period
        if not (prompt.endswith('.') or prompt.endswith('?') or prompt.endswith('!')):
            prompt += '.'

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Only make API call if there is a prompt
        if prompt:

            # Create full prompt
            full_prompt = build_prompt(st.session_state['previous_interactions'], prompt + prompt_addition)
            
            # Set up counters so app tries request max_attempts, in case gpt returns bad code
            max_attempts = 5
            attempts = 0
            success = False
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):

                # Keep trying until max attempts
                while attempts < max_attempts and not success:
                    try:
    
                        # Make a request to the OpenAI API
                        response = make_api_call(full_prompt)
                        response = response.replace("```python", "")
                        response = response.replace("```", "")

                        # Move reults explanation to the end
                        response = move_explanation(response)
    
                        # Execute the script
                        exec(response, exec_globals)

                        # Print the script from GPT
                        with st.expander("Python Script"):
                            st.code(response, language='python')

                        # Update previous interactions with the latest response
                        st.session_state['previous_interactions'] += "\nUser: " + prompt + prompt_addition + "\nGPT: " + response
                        
                        # Set success if no errors
                        success = True
    
                    except Exception as e:
                        # st.error(f"An error occurred: {e}")
                        attempts += 1
    
                # Requests a different query if gpt keeps giving bad code
                if not success:
                    response = "Please try a different query."
                    st.write(response)
    
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Reset conversation button
    if st.button("Restart Conversation"):
        st.session_state['previous_interactions'] = ""
        st.session_state.messages = []
        df = df_saved.copy()
        st.experimental_rerun()

# Display dictionary page
def display_dictionary(dictionary):

    # Header
    st.header("Dictionary")

    # Write dictionary
    dictionary_copy = dictionary.copy()
    dictionary_copy['Definition'] = dictionary_copy['Definition'].str.replace('\n', '<br>', regex=False)
    html = dictionary_copy.to_html(index=False, escape=False)
    html = html.replace('<thead>', '<thead style="text-align: left;">')
    html = html.replace('<th>', '<th style="text-align: left;">')
    st.markdown(html, unsafe_allow_html=True)

# Display user guide page
def display_user_guide():

    # Header
    st.header("User Guide")

    # Put general description of app
    st.write("""This application can be used to query the SBA 504 historical performance data. This data is publically available and 
                furnished quarterly by the SBA. The data we have is as of September 2023. So far we only have data for originations
                since 2010. The underlying data is a random sample of monthly dynamic data. Example queries include asking for historical CDR's and CPR's,
                restricting to different populations.""")

    # Write tips for best query writing
    st.markdown("""
    **Tips For Writing Prompts**
    1. Write in full sentances
    2. Use exact variable names and values from the dictionary  
    3. Be as specific in your request as you can
    4. Remember that the code is dealing with a table  
    5. If you don't get what you want initially, try resubmitting the query

    **Example Queries**
    1. Plot the CDR by Date
    2. Plot the CPR by Loan Age
    3. Plot the CPR by Loan Age for the different BusinessType's. Round Loan Age by 12.  
    4. Plot the CDR by Loan Age when the Date was between 2015 and 2018. Restrict to where the record count is greater than 1500. Please also plot the record count by Loan Age on the secondary axis as bars. 
    5. Plot the model vs actual CPR by Date
    6. Get the model and actual CPR curves by Incentive for when Date was in 2016 and when Date was in 2023. Round Incentive to the nearest
                .25. Restrict to where Loan Age is between 60 and 84. Plot all four curves on the same graph.
    7. Plot the model vs actual CPR by Loan Age for loans where “Dentist” is in the NaicsDescription. Round Loan Age to the nearest 6.
    8. Plot the model vs actual CPR by BusinessAge as a bar chart.
    9. Plot CDR by Orig Market Rate. Round Orig Market Rate to the nearest 0.5.
    """)

# Submit query to gpt
def make_api_call(user_prompt):
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
            if line.strip().endswith(')') or line.strip().endswith('")'):
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

if __name__ == "__main__":
    main()
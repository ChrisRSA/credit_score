import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import datetime as dt

with open('models/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

le = LabelEncoder()
le.classes_= np.load('models/classes.npy', allow_pickle=True)

def cleaning(data):
    for i in range(3, 8):
        data.iloc[:, i] = pd.to_datetime(data.iloc[:, i], format="%b-%y")
    data['years_data'] = data.iloc[:, 3:8].max(axis=1).dt.year + 1 - data.iloc[:, 3:8].min(axis=1).dt.year
    data['has_payment_note'] = np.where(data.payment_note_date.notnull(), 1, 0)
    data.drop(columns=['company_id', 'payment_note_date', 'financials_date', 'financials_date-1', 'financials_date-2', 'financials_date-3', 'financials_date-4'], inplace=True)
    col = data.pop('has_payment_note')
    data.insert(0, 'has_payment_note', col)
    col = data.pop('years_data')
    data.insert(2, 'years_data', col)
    clean_data = data.fillna(0)
    return clean_data   


def run():
    
    from PIL import Image
    image = Image.open('images/logo.jpeg')
    image_hospital = Image.open('images/banner-img.png')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict customer credit scores')
    st.sidebar.success('https://explore-utilities.com')
    
    st.sidebar.image(image_hospital)

    st.title("Credit Score Prediction App")

    if add_selectbox == 'Online':

        has_payment_note = st.number_input('Has payment note', min_value=0, max_value=1, value=0, key=1)
        payment_note_amount = st.number_input('Number of Years', min_value=1, max_value=5, value=5, key=2)
        years_data = st.number_input('Number of Years', min_value=1, max_value=5, value=5, key=3)
        revenue = st.number_input('Revenue', min_value=-1000, max_value=1000000, value=0, key=4)
        revenue_1 = st.number_input('Revenue - 1', min_value=-1000, max_value=1000000, value=0, key=5)
        revenue_2 = st.number_input('Revenue - 2', min_value=-1000, max_value=1000000, value=0, key=6)
        revenue_3 = st.number_input('Revenue - 3', min_value=-1000, max_value=1000000, value=0, key=7)
        revenue_4 = st.number_input('Revenue - 4', min_value=-1000, max_value=1000000, value=0, key=8)
        net_sales = st.number_input('Net Sales', min_value=-1000, max_value=1000000, value=0, key=9)
        net_sales_1 = st.number_input('Net Sales - 1', min_value=-1000, max_value=1000000, value=0, key=10)
        net_sales_2 = st.number_input('Net Sales - 2', min_value=-1000, max_value=1000000, value=0, key=11)
        net_sales_3 = st.number_input('Net Sales - 3', min_value=-1000, max_value=1000000, value=0, key=12)
        net_sales_4 = st.number_input('Net Sales - 4', min_value=-1000, max_value=1000000, value=0, key=13)
        panfi = st.number_input('Profit after net financial items', min_value=-1000, max_value=1000000, value=0, key=14)
        panfi_1 = st.number_input('Profit after net financial items - 1', min_value=-1000, max_value=1000000, value=0, key=15)
        panfi_2 = st.number_input('Profit after net financial items - 2', min_value=-1000, max_value=1000000, value=0, key=16)
        panfi_3 = st.number_input('Profit after net financial items - 3', min_value=-1000, max_value=1000000, value=0, key=17)
        panfi_4 = st.number_input('Profit after net financial items - 4', min_value=-1000, max_value=1000000, value=0, key=18)
        profit = st.number_input('Profit', min_value=-1000, max_value=1000000, value=0, key=19)
        profit_1 = st.number_input('Profit - 1', min_value=-1000, max_value=1000000, value=0, key=20)
        profit_2 = st.number_input('Profit - 2', min_value=-1000, max_value=1000000, value=0, key=21)
        profit_3 = st.number_input('Profit - 3', min_value=-1000, max_value=1000000, value=0, key=22)
        profit_4 = st.number_input('Profit - 4', min_value=-1000, max_value=1000000, value=0, key=23)
        assets = st.number_input('Assets', min_value=-1000, max_value=1000000, value=0, key=24)
        assets_1 = st.number_input('Assets - 1', min_value=-1000, max_value=1000000, value=0, key=25)
        assets_2 = st.number_input('Assets - 2', min_value=-1000, max_value=1000000, value=0, key=26)
        assets_3 = st.number_input('Assets - 3', min_value=-1000, max_value=1000000, value=0, key=27)
        assets_4 = st.number_input('Assets - 4', min_value=-1000, max_value=1000000, value=0, key=28)
        lt_liabilities = st.number_input('Long term liabilities', min_value=-1000, max_value=1000000, value=0, key=29)
        lt_liabilities_1 = st.number_input('Long term liabilities - 1', min_value=-1000, max_value=1000000, value=0, key=30)
        lt_liabilities_2 = st.number_input('Long term liabilities- 2', min_value=-1000, max_value=1000000, value=0, key=31)
        lt_liabilities_3 = st.number_input('Long term liabilities - 3', min_value=-1000, max_value=1000000, value=0, key=32)
        lt_liabilities_4 = st.number_input('Long term liabilities - 4', min_value=-1000, max_value=1000000, value=0, key=33)
        equity = st.number_input('Equity', min_value=-1000, max_value=1000000, value=0, key=34)
        equity_1 = st.number_input('Equity - 1', min_value=-1000, max_value=1000000, value=0, key=35)
        equity_2 = st.number_input('Equity - 2', min_value=-1000, max_value=1000000, value=0, key=36)
        equity_3 = st.number_input('Equity - 3', min_value=-1000, max_value=1000000, value=0, key=37)
        equity_4 = st.number_input('Equity - 4', min_value=-1000, max_value=1000000, value=0, key=38)
        equity_ratio = st.number_input('Equity', min_value=-1000, max_value=1000000, value=0, key=39)
        equity_ratio_1 = st.number_input('Equity Ratio - 1', min_value=-1000, max_value=1000000, value=0, key=40)
        equity_ratio_2 = st.number_input('Equity Ratio - 2', min_value=-1000, max_value=1000000, value=0, key=41)
        equity_ratio_3 = st.number_input('Equity Ratio - 3', min_value=-1000, max_value=1000000, value=0, key=42)
        equity_ratio_4 = st.number_input('Equity Ratio - 4', min_value=-1000, max_value=1000000, value=0, key=43)
        profit_margin = st.number_input('Profit Margin', min_value=-1000, max_value=1000000, value=0, key=44)
        profit_margin_1 = st.number_input('Profit Margin - 1', min_value=-1000, max_value=1000000, value=0, key=45)
        profit_margin_2 = st.number_input('Profit Margin - 2', min_value=-1000, max_value=1000000, value=0, key=46)
        profit_margin_3 = st.number_input('Profit Margin - 3', min_value=-1000, max_value=1000000, value=0, key=47)
        profit_margin_4 = st.number_input('Profit Margin - 4', min_value=-1000, max_value=1000000, value=0, key=48)
        cash_ratio = st.number_input('Cash Ratio', min_value=-1000, max_value=1000000, value=0, key=49)
        cash_ratio_1 = st.number_input('Cash Ratio - 1', min_value=-1000, max_value=1000000, value=0, key=50)
        cash_ratio_2 = st.number_input('Cash Ratio - 2', min_value=-1000, max_value=1000000, value=0, key=51)
        cash_ratio_3 = st.number_input('Cash Ratio - 3', min_value=-1000, max_value=1000000, value=0, key=52)
        cash_ratio_4 = st.number_input('Cash Ratio - 4', min_value=-1000, max_value=1000000, value=0, key=53)

        output=""

        input_dict = {
            'Has payment note': has_payment_note, 'Payment note amount': payment_note_amount,
            'Number of years': years_data, 'Revenue': revenue, 'Revenue - 1': revenue_1, 'Revenue - 2': revenue_2, 'Revenue - 3': revenue_3, 'Revenue - 4': revenue_4, 
            'Net Sales': net_sales, 'Net Sales - 1': net_sales_1, 'Net Sales - 2': net_sales_2, 'Net Sales - 3': net_sales_3, 'Net Sales - 4': net_sales_4,
            'Profit after net financial items': panfi , 'Profit after net financial items - 1': panfi_1, 'Profit after net financial items - 2': panfi_2,
            'Profit after net financial items - 3': panfi_3, 'Profit after net financial items - 4': panfi_4,
            'Profit': profit, 'Profit - 1': profit_1, 'Profit - 2': profit_2, 'Profit - 3': profit_3, 'Profit - 4': profit_4,
            'Assets': assets, 'Assets - 1': assets_1, 'Assets - 2': assets_2, 'Assets - 3': assets_3, 'Assets - 4': assets_4,
            'Long term liabilities': lt_liabilities, 'Long term liabilities -1': lt_liabilities_1, 'Long term liabilities - 2': lt_liabilities_2,
            'Long term liabilities - 3': lt_liabilities_3, 'Long term liabilities - 4': lt_liabilities_4,
            'Equity': equity, 'Equity - 1': equity_1, 'Equity - 2': equity_2, 'Equity - 3': equity_3, 'Equity - 4': equity_4,
            'Equity Ratio': equity_ratio, 'Equity Ratio - 1': equity_ratio_1, 'Equity Ratio - 2': equity_ratio_2, 'Equity Ratio - 3': equity_ratio_3, 'Equity Ratio - 4': equity_ratio_4,
            'Profit Margin': profit_margin, 'Profit Margin - 1': profit_margin_1, 'Profit Margin - 2': profit_margin_2, 'Profit Margin -3': profit_margin_3, 'Profit Margin - 4': profit_margin_4,
            'Cash Ratio': cash_ratio, 'Cash Ratio - 1': cash_ratio_1, 'Cash Ratio - 2': cash_ratio_2, 'Cash Ratio - 3': cash_ratio_3, 'Cash Ratio - 4': cash_ratio_4
            }
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = le.inverse_transform(model.predict(input_df))
            output = str(output).lstrip("['").rstrip("']")

        st.success('The output credit score is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            data = cleaning(data)
            predictions = le.inverse_transform(model.predict(data))
            st.write(predictions)

if __name__ == '__main__':
    run()
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from fpdf import FPDF

# tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
# tab1.write("this is tab 1")
# tab2.write("this is tab 2")

# st.file_uploader('Upload a CSV')

# uploaded_file = st.file_uploader('Upload a CSV')

# if uploaded_file is not None:
#     dataframe = pd.read_csv(uploaded_file)

#     st.dataframe(dataframe)

#     csv = dataframe.to_csv()

#     st.download_button(
#         label="Download CSV",
#         data=csv,
#         file_name='example.csv',
#         mime='text/csv',
#     )

# with open("dummy.pdf", "rb") as pdf_file:
#     PDFbyte = pdf_file.read()

# PDFbyte = "HI"

# csv = 'large_df.csv'

# st.download_button(label="Export_Report",
#                     data=PDFbyte,
#                     file_name="test.pdf",
#                     mime='application/octet-stream')

# st.download_button(
#     label="Download data as CSV",
#     data=csv,
#     file_name='large_df.csv',
#     mime='text/csv',
# )

data = []
Rd = pd.DataFrame(data, columns=['DIABETES', 'HEART DISEASE', 'LIVER DISEASE', 'KIDNEY DISEASE'])
print(Rd)

data = [['YES', 'NO', 'YES', 'NO']]

td = pd.DataFrame(data, columns=['DIABETES', 'HEART DISEASE', 'LIVER DISEASE', 'KIDNEY DISEASE'])
Rd = pd.concat([Rd,td], ignore_index = True)

print(Rd)
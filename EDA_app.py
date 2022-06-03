import streamlit as st
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

st.header('`streamlit_pandas_profiling`')

df = pd.read_csv('https://raw.githubusercontent.com/dfdavila2/TFM_2022_dfdavila/main/cleaned_data.csv?token=GHSAT0AAAAAABQBEW4C5HXFH4ZLMQRPFSESYUO6KHQ')

# https://github.com/dfdavila2/TFM_2022_dfdavila/blob/main/cleaned_data.csv')

pr = df.profile_report()
st_profile_report(pr)

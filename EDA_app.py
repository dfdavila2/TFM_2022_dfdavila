import streamlit as st
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

st.header('`Análisis exploratorio de los datos`')

@st.cache(suppress_st_warning=True)  # 👈 Changed this

def load_data():
  df = pd.read_csv('https://raw.githubusercontent.com/dfdavila2/TFM_2022_dfdavila/main/cleaned_data.csv?token=GHSAT0AAAAAABQBEW4C5HXFH4ZLMQRPFSESYUO6KHQ')
  return df

# https://github.com/dfdavila2/TFM_2022_dfdavila/blob/main/cleaned_data.csv')

pr = load_data().profile_report()

# pr = df.profile_report(title="Pandas Profiling Report", explorative=True)

# profile = ProfileReport(df, )

st_profile_report(pr)

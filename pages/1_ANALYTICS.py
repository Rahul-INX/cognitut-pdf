import streamlit as st
import pandas as pd
import numpy as np
from streamlit_gsheets import GSheetsConnection
import matplotlib.pyplot as plt

st.set_page_config(page_title="ANALYTICS", page_icon="📈")

st.markdown("# :red[ INTERACTION ANALYTICS 📈]")
st.sidebar.header("MENU")

analytics = st.sidebar.radio("**CHOOSE FROM THE OPTIONS** :", ["*View Interactions*", "*Query Analytics*", "*Cognitive Analytics*", "*Summary*"], index=0)

conn = st.connection('gsheets', type=GSheetsConnection)
database = conn.read(worksheet='Sheet1', usecols=list(range(16)), ttl=0)

# Convert the database to a DataFrame
df = pd.DataFrame(database)

if analytics == "*View Interactions*":
    st.write(df)

elif analytics == "*Query Analytics*":
    # Convert 'date_time' to datetime format
    df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')

    # Drop rows where 'date_time' is NaT
    df.dropna(subset=['date_time'], inplace=True)

    # Initialize an empty DataFrame to store daily counts
    daily_counts = pd.DataFrame(index=df['date_time'].dt.date.unique()).sort_index()

    # Loop through each feature to count the occurrences of 1 per day
    for col in ["explicit", "discrimination", "academic absence", "harmful intent"]:
        daily_counts[col] = df.groupby(df['date_time'].dt.date)[col].apply(lambda x: np.sum(x == 1))

    # Plot the counts against time using a line chart
    st.line_chart(daily_counts)

elif analytics == "*Cognitive Analytics*":
    # Clean and count occurrences of specified values in "blooms classification"
    df["blooms classification"] = df["blooms classification"].str.upper().str.replace(r'[^K1K2K3K4K5K6]', '')

    # Count occurrences of specified values
    blooms_counts = df["blooms classification"].value_counts().reindex(['K1', 'K2', 'K3', 'K4', 'K5', 'K6']).fillna(0)

    # Plot the counts on a bar chart
    st.bar_chart(blooms_counts)

elif analytics == "*Summary*":
    # Total Interactions
    total_interactions = len(df)
    st.write(f"## :green[Total Interactions: {total_interactions}]")

    # Pie chart for k1 to k6
    blooms_counts = df["blooms classification"].str.upper().str.replace(r'[^K1K2K3K4K5K6]', '')
    blooms_counts = blooms_counts.value_counts().reindex(['K1', 'K2', 'K3', 'K4', 'K5', 'K6']).fillna(0)
    
    fig1, ax1 = plt.subplots()
    ax1.pie(blooms_counts, labels=blooms_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write("## **Pie chart for k1 to k6 :**")
    st.pyplot(fig1)
  

    # Pie chart for doc_mode
    df['doc_mode'] = df['doc_mode'].apply(lambda x: 1 if x == 1 else 0)
    
    doc_mode_counts = df['doc_mode'].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.pie(doc_mode_counts, labels=doc_mode_counts.index, autopct='%1.1f%%', startangle=90)
    ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write("## **Pie chart for doc_mode:**")
    st.pyplot(fig3)

    # Pie chart for semester
    semester_counts = df['semester'].value_counts()
    fig4, ax4 = plt.subplots()
    ax4.pie(semester_counts, labels=semester_counts.index, autopct='%1.1f%%', startangle=90)
    ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write("## **Pie chart for semester:**")
    st.pyplot(fig4)

    # Pie chart for department
    department_counts = df['department'].value_counts()
    fig5, ax5 = plt.subplots()
    ax5.pie(department_counts, labels=department_counts.index, autopct='%1.1f%%', startangle=90)
    ax5.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write("## **Pie chart for department:**")
    st.pyplot(fig5)

    # Pie chart for llm
    llm_counts = df['llm'].value_counts()
    fig6, ax6 = plt.subplots()
    ax6.pie(llm_counts, labels=llm_counts.index, autopct='%1.1f%%', startangle=90)
    ax6.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write("## **Pie chart for llm:**")
    st.pyplot(fig6)

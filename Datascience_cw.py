import streamlit as st
import plotly.express as px
import pandas as pd
import warnings
import plotly.graph_objects as go 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Your Superstore Insights", page_icon="📊", layout="wide")

st.title("Dashboard - Superstore")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# read csv
default_path = r"C:\Users\Umangi\ICW\Global_Superstore_Lite_Origi.csv"
df = pd.read_csv(default_path, encoding="ISO-8859-1")

# Calculate KPIs
total_sales = df['Sales'].sum()
quantity_sold = df['Quantity'].sum()
average_profit = df['Profit'].mean()

# Display KPIs in cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Sales", value=f"${total_sales:,.2f}", delta=None)
with col2:
    st.metric(label="Quantity Sold", value=quantity_sold, delta=None)
with col3:
    st.metric(label="Average Profit", value=f"${average_profit:,.2f}", delta=None)

# Sidebar filters
st.sidebar.header("Filter Your Data:")
region = st.sidebar.multiselect("Pick Regions", df["Region"].unique())
state = st.sidebar.multiselect("Pick States", df["State"].unique())
city = st.sidebar.multiselect("Pick Cities", df["City"].unique())

startDate = pd.to_datetime(df["Order Date"]).min()
endDate = pd.to_datetime(df["Order Date"]).max()

with st.sidebar.expander("Date Range"):
    date1 = pd.to_datetime(st.sidebar.date_input("Start Date", startDate))
    date2 = pd.to_datetime(st.sidebar.date_input("End Date", endDate))

# Apply filters
if not region and not state and not city:
    filtered_df = df
else:
    filtered_df = df[df["Region"].isin(region) & df["State"].isin(state) & df["City"].isin(city) &
                     (df["Order Date"] >= date1) & (df["Order Date"] <= date2)]

def generate_choropleth_map(data, locations_col, color_col, title):
    fig = px.choropleth(
        data,
        locations=locations_col,
        color=color_col,
        locationmode='country names',
        color_continuous_scale='Viridis',
        labels={color_col: 'Sales'},
        title=title,
    )
    fig.update_geos(showcoastlines=True)
    return fig

# Region wise Sales
st.subheader("Region wise Sales")
choropleth_data = df.groupby('Country')['Sales'].sum().reset_index()
fig = generate_choropleth_map(choropleth_data, 'Country', 'Sales', 'Worldwide Sales')
fig.update_layout(height=700, width=1000) 
st.plotly_chart(fig, use_container_width=True)

def generate_pie_chart(data, values_col, names_col, title, template):
    fig = px.pie(data, values=values_col, names=names_col, template=template)
    fig.update_traces(text=data[names_col], textposition="inside")
    return fig

def download_data(data, columns, filename):
    data_to_download = data[columns]
    csv = data_to_download.to_csv(index=False).encode('utf-8')
    st.download_button(f"Download {filename} Data", data=csv, file_name=f"{filename}_Sales.csv", mime="text/csv")

def generate_pie_chart(data, values_col, names_col, title, template):
    fig = px.pie(data, values=values_col, names=names_col, template=template)
    fig.update_traces(text=data[names_col], textposition="inside")
    return fig

def download_data(data, columns, filename):
    data_to_download = data[columns]
    csv = data_to_download.to_csv(index=False).encode('utf-8')
    st.download_button(f"Download {filename} Data", data=csv, file_name=f"{filename}_Sales.csv", mime="text/csv")

# Segment wise Sales
col1, col2 = st.columns(2)

with col1:
    st.subheader('Segment wise Sales')
    fig_segment = generate_pie_chart(df, "Sales", "Segment", "Segment wise Sales", "plotly_dark")
    st.plotly_chart(fig_segment, use_container_width=True)
    st.button("View Segment Data", on_click=lambda: download_data(df, ['Segment', 'Sales'], "Segment"))

with col2:
    st.subheader('Category wise Sales')
    fig_category = generate_pie_chart(df, "Sales", "Category", "Category wise Sales", "gridon")
    st.plotly_chart(fig_category, use_container_width=True)
    st.button("View Category Data", on_click=lambda: download_data(df, ['Category', 'Sales'], "Category"))

def generate_top_n_products_chart(df, n, ascending=True):
    df_grouped = df.groupby('Product Name')['Sales'].sum().reset_index()
    df_grouped_sorted = df_grouped.sort_values(by='Sales', ascending=ascending).head(n)
    fig = px.bar(df_grouped_sorted, x='Product Name', y='Sales', title=f"Top {n} {'Most' if ascending else 'Least'} Sold Products")
    return fig, df_grouped_sorted

def download_data(data, filename):
    csv_data = data.to_csv(index=False).encode('utf-8')
    st.download_button(f"Download {filename} Data", data=csv_data, file_name=f"{filename}.csv", mime="text/csv")

# Top 3 Most Sold Products
top_n_most_chart, top_n_most_data = generate_top_n_products_chart(df, n=3)
# Top 3 Least Sold Products
least_chart, least_data = generate_top_n_products_chart(df, n=3, ascending=False)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 3 Most Sold Products")
    st.plotly_chart(top_n_most_chart, use_container_width=True)

    with st.expander("View/Download Top 3 Most Sold Products Data"):
        st.write(top_n_most_data)
        download_data(top_n_most_data, "Top_3_Most_Sold_Products")

with col2:
    st.subheader("Top 3 Least Sold Products")
    st.plotly_chart(least_chart, use_container_width=True)

    with st.expander("View/Download Top 3 Least Sold Products Data"):
        st.write(least_data)
        download_data(least_data, "Top_3_Least_Sold_Products")

def generate_time_series_chart(df):
    df["Order Date"] = pd.to_datetime(df["Order Date"]) 
    df["month_year"] = df["Order Date"].dt.to_period("M")
    linechart = df.groupby(df["month_year"].dt.strftime("%Y-%b"))["Sales"].sum().reset_index()
    fig = px.line(linechart, x="month_year", y="Sales", labels={"Sales": "Amount"}, height=500, width=1000, template="plotly_dark")
    return fig, linechart

def download_data(data, filename):
    csv_data = data.to_csv(index=False).encode("utf-8")
    st.download_button(f"Download {filename} Data", data=csv_data, file_name=f"{filename}.csv", mime="text/csv")

# Time Series Analysis
st.subheader('Time Series Analysis')
time_series_chart, time_series_data = generate_time_series_chart(df)
st.plotly_chart(time_series_chart, use_container_width=True)

with st.expander("View/Download Time Series Data"):
    st.write(time_series_data)
    download_data(time_series_data, "TimeSeries")

def generate_monthly_subcategory_table(df):
    df["month"] = df["Order Date"].dt.month_name()  
    sub_category_year = pd.pivot_table(data=df, values="Sales", index=["Sub-Category"], columns="month")
    return sub_category_year

# Define all sub-categories as a list
sub_categories = ['Accessories', 'Appliances', 'Binders', 'Bookcases', 'Chairs', 'Copiers', 'Furnishings', 'Machines', 'Phones', 'Storage', 'Supplies', 'Tables']

def generate_sample_data():
    data = pd.DataFrame({
        'Accessories': [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
        'Appliances': [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
        'Binders': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'Bookcases': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        'Chairs': [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
        'Copiers': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Furnishings': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Machines': [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        'Phones': [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        'Storage': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        'Supplies': [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'Tables': [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    })
    return data

st.subheader("Co-occurrence Heatmap")

data = generate_sample_data()

# Co-occurrence matrix 
co_occurrence_matrix = data.corr(method='spearman')  

# Set diagonal elements to 0 
np.fill_diagonal(co_occurrence_matrix.values, 0)

# Display heatmap 
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(co_occurrence_matrix, ax=ax, annot=data.values, cmap="YlGnBu", fmt="", linewidths=2)
ax.set_title('Co-occurrence Matrix of Sub-Categories (Excluding Same Product Combinations)')
ax.set_xlabel('Sub-Category (Consequent)')
ax.set_ylabel('Sub-Category (Antecedent)')
plt.xticks(ticks=np.arange(len(sub_categories)) + 0.5, labels=sub_categories, rotation=45)
plt.yticks(ticks=np.arange(len(sub_categories)) + 0.5, labels=sub_categories, rotation=0)
plt.tight_layout()
st.pyplot(fig)

# Month wise Sub-Category Sales
st.subheader('Month wise Sub-Category Sales')

month_wise_sub_category_sales = generate_monthly_subcategory_table(df)
st.write(month_wise_sub_category_sales.style.background_gradient(cmap="Greens"))

# View original Data
with st.expander("View Original Data"):
    st.write(filtered_df.iloc[:500, 1:20:2].style.background_gradient(cmap="Oranges"))

# Download original DataSet
csv = df.to_csv(index=False).encode('utf-8')
st.download_button('Download Original Data', data=csv, file_name="Original_Data.csv", mime="text/csv")

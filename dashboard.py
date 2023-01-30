import folium
import pandas as pd
import streamlit as st
import geopandas
import numpy as np
import plotly.express as px

from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

from datetime import datetime

st.set_page_config( layout='wide' )

#conversions



@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

    return data

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)

    return geofile

def set_feature(data):
    # add new features
    data['price_m2'] = data['price'] / data['sqft_lot']

    return data

def overview_data(data):
    # respostas das questões 1 e 2
    f_attributes = st.sidebar.multiselect('Entre com a coluna', data.columns)
    f_zipcode = st.sidebar.multiselect('Entre com código postal', data['zipcode'].unique())

    if (f_zipcode != []) & (f_attributes != []):
        df = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
    elif (f_zipcode != []) & (f_attributes == []):
        df = data.loc[data['zipcode'].isin(f_zipcode), :]
    elif (f_zipcode == []) & (f_attributes != []):
        df = data.loc[:, f_attributes]
    else:
        df = data.copy()

    st.header('Data Overview')
    st.write(df.head())

    c1, c2 = st.columns((1, 1))

    # Merge - questão 3

    df_1 = df[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df_2 = df[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df_3 = df[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df_4 = df[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    merge_1 = pd.merge(df_1, df_2, on='zipcode', how='inner')
    merge_2 = pd.merge(merge_1, df_3, on='zipcode', how='inner')
    merge_final = pd.merge(merge_2, df_4, on='zipcode', how='inner')

    merge_final.columns = ['zipcode', 'Total de imóveis', 'Preço Médio', 'Tamanho Médio da sala', 'Preço / m2']

    c1.header('Average Values')
    c1.dataframe(merge_final, height=600)

    # statistics descritive - resposta da questão 4
    num_attributes = df.select_dtypes(include=['int64', 'float64'])

    min_ = num_attributes.apply(np.min)
    max_ = num_attributes.apply(np.max)
    std_ = num_attributes.apply(np.std)
    mean_ = num_attributes.apply(np.mean)
    median_ = num_attributes.apply(np.median)

    descritive = pd.concat([min_, max_, std_, mean_, median_], axis=1).reset_index()
    descritive.columns = ['Atributos', 'Mínimo', 'Máximo', 'Desvio Padrão', 'Média', 'Mediana']

    c2.header('Descriptive Analitics')
    c2.dataframe(descritive, height=600)

    return None

def portfolio_density(data, geofile):
    # Densidade de Portfolio - questão 5
    st.title('Region Overview')

    d1, d2 = st.columns((1, 1))

    d1.header('Portfolio Density')

    df_5 = data.sample(10)

    # Base Map - Folium
    density_map = folium.Map(location=[df_5['lat'].mean(),
                                       df_5['long'].mean()],
                             default_zoom_start=15)

    make_cluster = MarkerCluster().add_to(density_map)
    for name, row in df_5.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(
                          row['price'],
                          row['date'],
                          row['sqft_living'],
                          row['bedrooms'],
                          row['bathrooms'],
                          row['yr_built'])
                      ).add_to(make_cluster)

    with d1:
        folium_static(density_map)

    # Region Price Map
    d2.header('Price Density')

    df_6 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df_6.columns = ['ZIP', 'PRICE']
    df_6 = df_6.sample(20)

    geofile = geofile[geofile['ZIP'].isin(df_6['ZIP'].tolist())]

    region_price_map = folium.Map(
        location=[data['lat'].mean(),
                  data['long'].mean()],
        default_zoom_start=15
    )
    region_price_map.choropleth(
        data=df_6,
        geo_data=geofile,
        columns=['ZIP', 'PRICE'],
        key_on='feature.properties.ZIP',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='AVG PRICE'
    )
    with d2:
        folium_static(region_price_map)

    return None

def comercial_distribution(data):
    df = data.copy()
    # Distribuição dos imóveis por categorias comerciais
    st.sidebar.title('Commercial options')
    st.title('Commercial Attributes')

    # ---------- Average Price per Year - questão 6
    # filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built',
                                     min_year_built,
                                     max_year_built,
                                     min_year_built)

    st.header('Average Price per Year Built')

    # data selection
    df_7 = data.loc[data['yr_built'] < f_year_built]
    df_7 = df_7[['price', 'yr_built']].groupby('yr_built').mean().reset_index()

    fig = px.line(
        df_7,
        x='yr_built',
        y='price'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Average Price per Day - questão 7

    st.header('Average Price per Day')
    st.sidebar.subheader('Select Max Date')

    # filters
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Date', min_date, max_date, min_date)

    # data filtering
    df['date'] = pd.to_datetime(df['date'])

    df_8 = df.loc[df['date'] < f_date]
    df_8 = df_8[['price', 'date']].groupby('date').mean().reset_index()

    # plot
    fig = px.line(
        df_8,
        x='date',
        y='price'
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------- Histograma
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    # Filter
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    f_price = st.sidebar.slider(
        'Price',
        min_price,
        max_price,
        avg_price
    )

    # data filtering
    df_9 = data.loc[data['price'] < f_price]
    df_9 = df_9[['id', 'price']]

    # data plot
    fig = px.histogram(
        df_9,
        x='price',
        nbins=50
    )

    st.plotly_chart(
        fig, use_container_width=True
    )

    return None

def attributes_distribution(data):
    # Distribuição dos imóveis por categorias físicas
    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    # filter
    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms',
                                      sorted(set(data['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox('Max number bathrooms',
                                       sorted(set(data['bathrooms'].unique())))

    z1, z2 = st.columns(2)

    # House per bedrooms
    z1.header('Houses per bedrooms')
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    z1.plotly_chart(fig, use_container_width=True)

    # House per bathrooms
    z2.header('Houses per bathrooms')
    df = data[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    z2.plotly_chart(fig, use_container_width=True)

    # filters
    f_floors = st.sidebar.selectbox('Max number of floor',
                                    sorted(set(data['floors'].unique())))
    f_waterview = st.sidebar.checkbox('Only Houses with Water View')

    z1, z2 = st.columns(2)

    # House per floors
    z1.header('Houses per floor')
    df = data[data['floors'] < f_floors]
    fig = px.histogram(df, x='floors', nbins=19)
    z1.plotly_chart(fig, use_container_width=True)

    # House per water view
    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    fig = px.histogram(df, x='waterfront', nbins=10)
    z2.plotly_chart(fig, use_container_width=True)

    return None

if __name__ == '__main__':
    #ETL
    #Data extration
    path = 'dataset/kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data(path)
    geofile = get_geofile(url)

    #Transformation

    data = set_feature(data)

    overview_data(data)

    portfolio_density(data, geofile)

    comercial_distribution(data)

    attributes_distribution(data)
    #Load














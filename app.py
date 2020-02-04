
#first for github
import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def main():
    data = pd.read_csv("OnlineRetail.csv",encoding = 'unicode_escape')

    st.title('Costomer Lifetime Value')

#fuck

    import plotly.plotly as py

    import plotly.graph_objs as go



    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['InvoiceYearMonth'] = data['InvoiceDate'].map(lambda date : date.year * 100 + date.month)

    data['Revenue'] = data['UnitPrice'] * data['Quantity']




    data_revenue = data.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()
    data_revenue['MonthlyGrowth'] = data_revenue['Revenue'].pct_change()
    data_uk = data.query("Country == 'United Kingdom'").reset_index(drop = True)
    data_monthly_active = data_uk.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()
    data_uk_monthly_order = data_uk.groupby(['InvoiceYearMonth'])['Quantity'].sum().reset_index()
    data_first_purchase = data_uk.groupby('CustomerID')['InvoiceDate'].min().reset_index()
    data_first_purchase.columns = ['CustomerID','FirstPurchaseDate']
    data_first_purchase['FirstPurchaseYearMonth'] = data_first_purchase['FirstPurchaseDate'].map(
        lambda date: 100* date.year + date.month)
    data_uk_monthly_revenue_avg = data_uk.groupby(['InvoiceYearMonth'])['Revenue'].mean().reset_index()

    def plot_no_query(x,y, title, data = data_revenue):
        plot_data = [
            go.Scatter(
                x=data[x],
                y=data[y]
            )
        ]

        plot_layout = go.Layout(
            xaxis={"type": "category"},
            title=title
        )
        fig = go.Figure(data=plot_data, layout=plot_layout)

        return st.plotly_chart(fig)

    def plot_with_query(x,x_query,y,y_query, title, data = data_revenue):
        plot_data = [
            go.Scatter(
                x=data.query(x_query)[x],
                y=data.query(y_query)[y]
            )
        ]

        plot_layout = go.Layout(
            xaxis={"type": "category"},
            title=title
        )
        fig = go.Figure(data=plot_data, layout=plot_layout)

        return st.plotly_chart(fig)
    #%%
    data_uk = pd.merge(data_uk, data_first_purchase, on = 'CustomerID')
    data_uk['UserType'] = 'New'
    data_uk.loc[data_uk['InvoiceYearMonth'] > data_uk['FirstPurchaseYearMonth'],'UserType'] = 'Existing'

    data_usertype_revenue = data_uk.groupby(['InvoiceYearMonth','UserType'])['Revenue'].sum().reset_index()
    data_usertype_revenue = data_usertype_revenue.query('InvoiceYearMonth != 201012 and InvoiceYearMonth != 201112')
    data_user_rato = data_uk.query('UserType == "New"').groupby(['InvoiceYearMonth'])['CustomerID'].nunique() / data_uk.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()
    data_user_rato = data_user_rato.reset_index()
    data_user_rato.dropna()
    data_user_purchase = data_uk.groupby(['CustomerID','InvoiceYearMonth'])['Revenue'].sum().reset_index()

    data_retention = pd.crosstab(data_user_purchase['CustomerID'],data_user_purchase['InvoiceYearMonth']).reset_index()

    #%%
    activity = ["EDA", 'Matrix',"RFM describe", "Model"]
    choice = st.sidebar.selectbox('Select Activity', activity)

    if choice == 'EDA':
        st.header("Exploratory Data Analysis")
        if st.checkbox("Show Summary"):
            st.write(data.describe())
        if st.checkbox("First 10 rows"):
            st.write(data.head(10))
        if st.checkbox("Select columns to show"):
            selected_col = st.multiselect("Select Columns", data.columns.to_list())
            new_col = data[selected_col]
            st.dataframe(new_col)


    elif choice == 'Matrix':
        st.header("Matrix for Online Shopping")
        plot = st.radio("Select difference plots you want to explore",
                        ("Monthly Revenue", "Monthly Growth Rate", "Monthly Active",
                         "Monthly Order", "Monthly Order Average", "New vs Existing",
                         "Monthly User Ratio"))

        if plot == "Monthly Revenue":
            plot_no_query('InvoiceYearMonth', 'Revenue', 'Monthly Revenue')
        elif plot == 'Monthly Growth Rate':
            plot_with_query('InvoiceYearMonth', 'InvoiceYearMonth < 201112', 'MonthlyGrowth', 'InvoiceYearMonth < 201112',
                            'Monthly Growth Rate')
        elif plot == 'Monthly Active':
            plot_no_query('InvoiceYearMonth', 'CustomerID', 'Monthly Active', data_monthly_active)
        elif plot == 'Monthly Order':
            plot_no_query('InvoiceYearMonth', 'Quantity', 'Monthly Order', data_uk_monthly_order)
        elif plot == "Monthly Order Average":
            plot_no_query('InvoiceYearMonth', 'Revenue', "Monthly  Order Average", data_uk_monthly_revenue_avg)
        elif plot == "New vs Existing":
            plot_data = [
                go.Scatter(
                    x=data_usertype_revenue.query('UserType == "New"')['InvoiceYearMonth'],
                    y=data_usertype_revenue.query('UserType == "New"')['Revenue'],
                    name='New'
                ),
                go.Scatter(
                    x=data_usertype_revenue.query('UserType == "Existing"')['InvoiceYearMonth'],
                    y=data_usertype_revenue.query('UserType == "Existing"')['Revenue'],
                    name='Existing'
                )
            ]

            plot_layout = go.Layout(
                xaxis={"type": "category"},
                title='New vs Existing'
            )
            fig = go.Figure(data=plot_data, layout=plot_layout)

            st.plotly_chart(fig)
        elif plot == "Monthly User Ratio":
            plot_with_query('InvoiceYearMonth', "InvoiceYearMonth > 201101 and InvoiceYearMonth < 201112", 'CustomerID',
                            "InvoiceYearMonth > 201101 and InvoiceYearMonth < 201112", 'Monthly User Ratio', data_user_rato)

    elif choice == "RFM describe":
        st.header("Revenue, Frequency, Monetary Value describe")
        # %%
        # Customer Segmentation
        def Revenue_fun(data = data):
            data_user = pd.DataFrame(data['CustomerID'].unique())

            data_user.columns = ['CustomerID']
            data_max_purchase = data_uk.groupby('CustomerID')['InvoiceDate'].max().reset_index()
            data_max_purchase.columns = ['CustomerID', 'MaxPurchaseDate']
            data_max_purchase['Recency'] = (
                        data_max_purchase['MaxPurchaseDate'].max() - data_max_purchase['MaxPurchaseDate']).dt.days
            data_user = pd.merge(data_user, data_max_purchase[['CustomerID', 'Recency']], on='CustomerID')
            return data_user


        # %%
        def RFM_plot(data, column, title):
            plot_data = [
                go.Histogram(
                    x=data[column]
                )
            ]

            plot_layout = go.Layout(
                title=title
            )
            fig = go.Figure(data=plot_data, layout=plot_layout)

            st.plotly_chart(fig)

        Reveune = Revenue_fun()

        def order_cluser(cluster_field_name, target_field_name, df, ascending):
            new_cluster_field_name = 'new_' + cluster_field_name
            df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
            df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
            df_new['index'] = df_new.index
            df_final = pd.merge(df, df_new[[cluster_field_name, 'index']], on=cluster_field_name)
            df_final = df_final.drop([cluster_field_name], axis=1)
            df_final = df_final.rename(columns={'index': cluster_field_name})
            return df_final

        from sklearn.cluster import KMeans

        # %%
        data_recency = Reveune[['Recency']]
        sse = {}
        for i in range(1, 10):
            kmeans = KMeans(n_clusters=i, max_iter=1000, n_jobs=4).fit(data_recency)
            data_recency["clusters"] = kmeans.labels_
            sse[i] = kmeans.inertia_
        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()))
        plt.xlabel("Number of cluster")
        plt.show()

        kmeans = KMeans(n_clusters=4)
        kmeans.fit(Reveune[['Recency']])
        Reveune['RecencyCluster'] = kmeans.predict(Reveune[['Recency']])


        #%%
        data_user = order_cluser('RecencyCluster', "Recency", Reveune, False)


        #%%
        data_frequency = data_uk.groupby('CustomerID')['InvoiceDate'].count().reset_index()
        data_frequency.columns = ['CustomerID', 'Frequency']
        data_user = pd.merge(data_user, data_frequency, on="CustomerID")
        def Fre_plot():
            plot_data = [
                go.Histogram(
                    x=data_user.query('Frequency < 1000')['Frequency']
                )
            ]

            plot_layout = go.Layout(
                title='Frequnecy'
            )
            fig = go.Figure(data=plot_data, layout=plot_layout)

            st.plotly_chart(fig)

        # %%

        kmeans = KMeans(n_clusters=4)
        kmeans.fit(data_user[['Frequency']])
        data_user['FrequencyCluster'] = kmeans.predict(data_user[['Frequency']])
        data_user = order_cluser('FrequencyCluster', "Frequency", data_user, True)

        # %%
        data_uk['Revenue'] = data_uk['UnitPrice'] * data_uk['Quantity']
        data_revenue = data_uk.groupby('CustomerID').Revenue.sum().reset_index()
        data_user = pd.merge(data_user, data_revenue, on="CustomerID")
        def Reve_plot():
            plot_data = [
                go.Histogram(
                    x=data_user.query('Revenue < 10000')['Revenue']
                )
            ]

            plot_layout = go.Layout(
                title='Monetary Value'
            )
            fig = go.Figure(data=plot_data, layout=plot_layout)

            st.plotly_chart(fig)

        kmeans = KMeans(n_clusters=4)
        kmeans.fit(data_user[['Revenue']])
        data_user['RevenueCluster'] = kmeans.predict(data_user[['Revenue']])

        # order the cluster numbers
        data_user = order_cluser('RevenueCluster', 'Revenue', data_user, True)

        # show details of the dataframe



    #%%
        data_user['OverallScore'] = data_user['FrequencyCluster'] + data_user['RecencyCluster'] + data_user['RevenueCluster']
        data_user.groupby('OverallScore')['Recency', 'Frequency', 'Revenue'].mean()


        data_user['Segment'] = 'Low-Value'
        data_user.loc[data_user.OverallScore > 2, 'Segment'] = 'Mid-Value'
        data_user.loc[data_user.OverallScore > 4, 'Segment'] = 'High-Value'
        from datetime import date


        data_3m = data_uk[((data_uk.InvoiceDate < '2011-06-01') & (data_uk.InvoiceDate >= '2011-03-01'))].reset_index(
            drop=True)
        data_6m = data_uk[((data_uk.InvoiceDate >= '2011-06-01') & (data_uk.InvoiceDate < '2011-12-01'))].reset_index(
            drop=True)
        data_user_6m = data_6m.groupby('CustomerID')['Revenue'].sum().reset_index()
        data_user_6m.columns = ['CustomerID', 'Revenue_6m']

        data_merge = pd.merge(data_user, data_user_6m, on='CustomerID', how='left')
        data_merge = data_merge.fillna(0)

        data_graph = data_merge.query('Revenue_6m < 30000')

        def Segments_plot(x_label, y_label, data_graph = data_graph):
            data_graph = data_graph.query("Revenue < 50000 and Frequency < 2000")
            plot_data = [
                go.Scatter(
                    x=data_graph.query("Segment == 'Low-Value'")[x_label],
                    y=data_graph.query("Segment == 'Low-Value'")[y_label],
                    mode='markers',
                    name='Low',
                    marker=dict(size=7,
                                line=dict(width=1),
                                color='blue',
                                opacity=0.8
                                )
                ),
                go.Scatter(
                    x=data_graph.query("Segment == 'Mid-Value'")[x_label],
                    y=data_graph.query("Segment == 'Mid-Value'")[y_label],
                    mode='markers',
                    name='Mid',
                    marker=dict(size=9,
                                line=dict(width=1),
                                color='green',
                                opacity=0.5
                                )
                ),
                go.Scatter(
                    x=data_graph.query("Segment == 'High-Value'")[x_label],
                    y=data_graph.query("Segment == 'High-Value'")[y_label],
                    mode='markers',
                    name='High',
                    marker=dict(size=11,
                                line=dict(width=1),
                                color='red',
                                opacity=0.9
                                )
                ),
            ]

            plot_layout = go.Layout(
                yaxis={'title': y_label},
                xaxis={'title': x_label},
                title='Segments'
            )
            fig = go.Figure(data=plot_data, layout=plot_layout)
            st.plotly_chart(fig)
        if st.checkbox("Frequency"):
            Fre_plot()
            st.write(data_user.groupby('FrequencyCluster')['Frequency'].describe())
            Segments_plot('Frequency', 'Revenue')
        if st.checkbox("Recency"):
            RFM_plot(Reveune, 'Recency', 'Recency')
            st.write(data_user.groupby('RecencyCluster')['Recency'].describe())
            Segments_plot('Recency', 'Revenue')
        if st.checkbox("Revenue"):
            Reve_plot()
            st.write(data_user.groupby('RevenueCluster')['Revenue'].describe())
            Segments_plot('Recency', 'Frequency')
    #%%
    elif choice == "Model":
        st.subheader("Model")




if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np

### Config
st.set_page_config(
    page_title="GetAround analysis",
    page_icon="🚙",
    layout="centered"
)

### Data upload
DATA_URL = ("https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx")

@st.cache
def load_data():
    data = pd.read_excel(DATA_URL)
    return data

data = load_data()
# Selecting data with delay info
data_complete = data.loc[data["time_delta_with_previous_rental_in_minutes"].notna(),:]

### Setting personalised palette
purples = ['#F6E5F5', '#CA6EC3', '#C04FB8', '#B01AA7', '#8D1586', '#5F1159']
pio.templates["purples"] = go.layout.Template(
    layout = {
        'title':
            {'font': {'color': '#0F0429'}
            },
        'font': {'color': '#0F0429'},
        'colorway': purples,
    }
)
pio.templates.default = "purples"

### Title and intro

st.markdown('<style>body{background-color: F1F1F4;}</style>',unsafe_allow_html=True)

st.image("https://lever-client-logos.s3.amazonaws.com/2bd4cdf9-37f2-497f-9096-c2793296a75f-1568844229943.png")
#st.title("GetAround rentals delay analysis")
st.markdown("<h1 style='text-align: center'>GetAround rentals delay analysis</h1>", unsafe_allow_html=True)


st.markdown("""Check-in delays are a reported cause of rentals cancellation. A minimum delay between two rentals can be applied and a vehicle won't be displayed in the search results.\n
To avoid revenue loss for the owner and GetAround, a right trade-off must be chosen, along with the appropriate revenue share to be applied to.
The main revenue shares are the connect and mobile check-ins. In the former the renter can access the vehicle without contact with the owner, in the latter a rental agreement is signed between the two on the owner's mobile phone.\n
The following overview of rental data can offer insight in rental delays and the appropriate minumum delay to apply to search results.
""")

###Graph 1####################################################################################################
st.subheader("Rentals shares overview")

fig = px.sunburst(data, path=['checkin_type', 'state'],
                  color_discrete_sequence = ['#8D1586','#261A48']
)
fig.update_traces(textinfo="label+percent parent")
st.plotly_chart(fig, use_container_width=True)

###Graph 2####################################################################################################

st.subheader("Which revenue share is most affected by delays? ")

average_delay = data.groupby(['time_delta_with_previous_rental_in_minutes', 'checkin_type'])['delay_at_checkout_in_minutes'].mean().reset_index(level=1)
fig = px.line(average_delay,
              color="checkin_type",
              color_discrete_sequence=['#261A48', '#8D1586'],
              title ="Average delay at checkout as a function of time between rentals")

fig.update_xaxes(title_text="Time between rentals")
fig.update_yaxes(title_text="Delay")
fig.update_layout(
    legend_title_text='',
    legend=dict(
        bgcolor=None,
        x=0.82,
        y=1,
        font=dict(
            size=16,      
        )
    )
)
fig.update_layout(
    title=dict(
        font=dict(
            size=20,
        )
    ))

###Graph 3####################################################################################################

st.plotly_chart(fig, use_container_width=True)

plot_data=data_complete.loc[data["state"]=="ended",:]
plot_data['delay'] = plot_data['delay_at_checkout_in_minutes'].map(lambda x: "On time" if (x<=0 or np.isnan(x)) else "Delayed")

fig = px.sunburst(plot_data, path=['checkin_type', 'delay'],
                  color_discrete_sequence = ['#8D1586','#261A48'],
                 )

fig.update_traces(textinfo="label+percent parent")
st.plotly_chart(fig, use_container_width=True)

###Graph 4####################################################################################################

st.subheader("How does the delay of a driver impact the next driver?")

#Selecting rental ids for rentals that have a following rental 
previous_rentals = data.loc[~data['previous_ended_rental_id'].isna(),'previous_ended_rental_id'].values.tolist()
#Selecting the rentals and their following rental
previous_rental_data = data.loc[data['rental_id'].isin(previous_rentals)|data['previous_ended_rental_id'].isin(previous_rentals),:]

#Creating a binary delay column for rentals and their following rental
def next_rental_delayed(x):
    try:
        y = previous_rental_data.loc[previous_rental_data['previous_ended_rental_id']==x,'delay_at_checkout_in_minutes'].values[0]
        if np.isnan(y) or y<=0:
            return "On time"
        else:
            return "Delayed"
    except:
        return np.nan
#make a delay column
previous_rental_data['delay'] = previous_rental_data['delay_at_checkout_in_minutes'].map(lambda x: "On time" if (x<=0 or np.isnan(x)) else "Delayed")
#make a next_rental_delayed column  
previous_rental_data['delay_next_rental'] = previous_rental_data['rental_id'].map(lambda x:next_rental_delayed(x))

plot_data = previous_rental_data.loc[previous_rental_data['rental_id'].isin(previous_rentals),:]

fig = px.icicle(plot_data, path=[px.Constant("Rentals"),'checkin_type','delay', 'delay_next_rental'],
                title="Check-in type, First driver, Following driver",
                 )

fig.update_traces(root_color='#f6e5f5')
fig.update_traces(textinfo="label+percent parent")
fig.update_layout(
    title=dict(
        font=dict(
            size=20,
        )
    ))
fig.update_traces(
    sort=False)
    
st.plotly_chart(fig, use_container_width=True)

###Graph 5####################################################################################################

plot_data = previous_rental_data.loc[(previous_rental_data['rental_id'].isin(previous_rentals))&(previous_rental_data['delay']=='Delayed')&(previous_rental_data['delay_at_checkout_in_minutes']<=300),:]

fig = px.histogram(plot_data, 
                   nbins=12,
                   barnorm="percent",
                   text_auto=True,
                   x="delay_at_checkout_in_minutes",
                   color="delay_next_rental",
                  title="Outcome of the following rental as a function of the previous rentals delay" )

fig.update_layout(bargap=0.2)
fig.update_xaxes(title_text="Delay at check-in")
fig.update_yaxes(title_text="Riders")
fig.update_layout(
    title=dict(
        font=dict(
            size=20,
        )
    ))
fig.update_layout(
    legend_title_text='Following rental',
    legend=dict(
        font=dict(
            size=13,      
        )
    )
)
st.plotly_chart(fig, use_container_width=True)

###Graph 6####################################################################################################


st.subheader("How many rentals would be affected by the feature depending on the threshold and scope we choose?")

threshold = st.number_input('Insert display threshold in minutes:', min_value=0, max_value=740,key="key_1" ) 

st.write('Apply threshold on:')
mobile_opt = st.checkbox('Mobile', key="key_3")
connect_opt = st.checkbox('Connect', key="key_4")

plot_data=data_complete.loc[data['state']=="ended",:]

mobile_share = 0
connect_share = 0

mobile_affected = 0
connect_affected = 0

total_mobile = plot_data.loc[(plot_data["checkin_type"]=="mobile"),:]["rental_id"].count()
total_connect = plot_data.loc[(plot_data["checkin_type"]=="connect"),:]["rental_id"].count()


if mobile_opt and connect_opt:
    mobile_affected = plot_data.loc[(plot_data["time_delta_with_previous_rental_in_minutes"]<=threshold)&(plot_data["checkin_type"]=="mobile"),:]["rental_id"].count()
    connect_affected = plot_data.loc[(plot_data["time_delta_with_previous_rental_in_minutes"]<=threshold)&(plot_data["checkin_type"]=="connect"),:]["rental_id"].count()
elif mobile_opt and not connect_opt:
    mobile_affected = plot_data.loc[(plot_data["time_delta_with_previous_rental_in_minutes"]<=threshold)&(plot_data["checkin_type"]=="mobile"),:]["rental_id"].count()
elif not mobile_opt and connect_opt:
    connect_affected = plot_data.loc[(plot_data["time_delta_with_previous_rental_in_minutes"]<=threshold)&(plot_data["checkin_type"]=="connect"),:]["rental_id"].count()

mobile_share = mobile_affected/total_mobile
connect_share = connect_affected/total_connect
total_share = (mobile_affected+connect_affected)/(total_mobile+total_connect)

fig = px.histogram(plot_data, 
                   x="time_delta_with_previous_rental_in_minutes",
                   #nbins=15,
                   #barmode="overlay",
                   color="checkin_type",
                   #cumulative=True,
                   color_discrete_sequence=['#261A48', '#8D1586'])

fig.update_layout(bargap=0.2)
fig.add_vline(x=int(threshold), 
              line_dash = 'dot', line_color = '#0f0429', 
              annotation_text=f"Cut-off: {threshold} minutes <br><br>Mobile share lost: {round(mobile_share*100,0)}%  <br>Connect share lost: {round(connect_share*100,0)}%  <br>Total share lost: {round(total_share*100,0)}%", 
              annotation_position="top right",
              annotation_align="left",
              annotation_font_size=13,
              annotation_font_color="#0f0429" )

fig.update_layout(
    legend_title_text='',
    legend=dict(
        font=dict(
            size=16,      
        )
    )
)

fig.update_xaxes(title_text="Time between rentals")
fig.update_yaxes(title_text="Ended rentals")
st.plotly_chart(fig, use_container_width=True)

###Graph 7####################################################################################################

st.subheader(" How many problematic cases will it solve?")

threshold_2 = st.number_input('Insert display threshold in minutes:', min_value=0, max_value=740, key="key_2" ) 

st.write('Apply threshold on:')
mobile_opt = st.checkbox('Mobile', key="key_5")
connect_opt = st.checkbox('Connect', key="key_6")

plot_data=data_complete.loc[data['state']=="canceled",:]

mobile_share = 0
connect_share = 0

mobile_affected = 0
connect_affected = 0

total_mobile = plot_data.loc[(plot_data["checkin_type"]=="mobile"),:]["rental_id"].count()
total_connect = plot_data.loc[(plot_data["checkin_type"]=="connect"),:]["rental_id"].count()


if mobile_opt and connect_opt:
    mobile_affected = plot_data.loc[(plot_data["time_delta_with_previous_rental_in_minutes"]<=threshold_2)&(plot_data["checkin_type"]=="mobile"),:]["rental_id"].count()
    connect_affected = plot_data.loc[(plot_data["time_delta_with_previous_rental_in_minutes"]<=threshold_2)&(plot_data["checkin_type"]=="connect"),:]["rental_id"].count()
elif mobile_opt and not connect_opt:
    mobile_affected = plot_data.loc[(plot_data["time_delta_with_previous_rental_in_minutes"]<=threshold_2)&(plot_data["checkin_type"]=="mobile"),:]["rental_id"].count()
elif not mobile_opt and connect_opt:
    connect_affected = plot_data.loc[(plot_data["time_delta_with_previous_rental_in_minutes"]<=threshold_2)&(plot_data["checkin_type"]=="connect"),:]["rental_id"].count()

mobile_share = mobile_affected/total_mobile
connect_share = connect_affected/total_connect
total_share = (mobile_affected+connect_affected)/(total_mobile+total_connect)

fig = px.histogram(plot_data, 
                   x="time_delta_with_previous_rental_in_minutes",
                   color="checkin_type",
                   color_discrete_sequence=['#261A48', '#8D1586'])

fig.update_layout(bargap=0.2)
fig.add_vline(x=threshold_2, 
              line_dash = 'dot', line_color = '#0f0429', 
              annotation_text=f"Cut-off: {threshold_2} minutes <br><br>Mobile share: {round(mobile_share*100,0)}%  <br>Connect share: {round(connect_share*100,0)}%  <br>Total share solved: {round(total_share*100,0)}%", 
              annotation_position="top right",
              annotation_align="left",
              annotation_font_size=13,
              annotation_font_color="#0f0429" )

fig.update_layout(
    legend_title_text='',
    legend=dict(
        font=dict(
            size=16,      
        )
    )
)

fig.update_xaxes(title_text="Time between rentals")
fig.update_yaxes(title_text="Canceled rentals")
st.plotly_chart(fig, use_container_width=True)
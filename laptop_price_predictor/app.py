import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Predictor")

st.image('Laptop.jpg', caption='Choose your laptop configuration',  use_container_width=True)


col1, col2 = st.columns(2)
with col1:
    company = st.selectbox('Brand', df['Company'].unique())
    type = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    cpu = st.selectbox('CPU', df['cpu Brand'].unique())
    os = st.selectbox('Operating System', df['OS'].unique())

with col2:
    weight = st.number_input('Weight (kg)')
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('GPU', df['Gpu_Brand'].unique())

with st.expander("🛠️ Advanced Display Settings"):
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS', ['No', 'Yes'])
    screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0)
    resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])


# Display Summary of Selections Before Prediction
if st.checkbox("Show Summary"):
    st.markdown(f"""
    **Brand:** {company}  
    **Type:** {type}  
    **RAM:** {ram} GB  
    **Weight:** {weight} kg  
    **Screen Resolution:** {resolution}  
    """)


if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.success(f"💰 **Estimated Price:** ₹ {predicted_price:,}")
    st.balloons()  # Celebrate the prediction with balloons



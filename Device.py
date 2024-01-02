import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from PIL import Image


st.header("SHOPFY")
st.subheader("Top Deals (Android, ios, Windows, others)")

st.sidebar.text('''Welcome to Shopfy!
Get an estimated price for your item.
Kindly fill all necessary details.
Chat us up or send us a message.
We hope to hear from you soon!''')
what1 = Image.open("whatsapp.png")
gmail1 = Image.open("gmail.png")
st.sidebar.image(what1, caption = "0809543864", width = 100)
st.sidebar.image(gmail1, caption = "shopfy@co.uk.org", width = 120)
hide_streamlit_style = """ 
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

import streamlit as st
from PIL import Image

# Load your images
image_paths = ['phone.jpg', 'smartphone.jpg', 'laptop.jpg']

target_size = (300, 300)

# Create columns for each image
columns = st.columns(len(image_paths))

# Display each resized image in a separate column
for i, path in enumerate(image_paths):
    img = Image.open(path)
    resized_img = img.resize(target_size)
    columns[i].image(resized_img, use_column_width=True)


st.title("Fill in the necessary details to get the best price to sell your used item!")

Device = pd.read_csv("used_device_data.csv")
Device1 = Device[Device["main_camera_mp"] < 41.0]
Device2 = Device1[Device1["int_memory"] < 256.0]
Device2["selfie_camera_mp"] = Device2["selfie_camera_mp"].fillna(Device2["selfie_camera_mp"].mean())
Device2["ram"] = Device2["ram"].fillna(Device2["ram"].mean())
Device2["battery"] = Device2["battery"].fillna(Device2["battery"].mean())
Device2["weight"] = Device2["weight"].fillna(Device2["weight"].mean())
x = Device2.drop("normalized_used_price", axis = 1)
y = Device2["normalized_used_price"].values
X = pd.get_dummies(x, columns=["brand_name", "os", "4g", "5g"])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=24)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_trainsc = sc.fit_transform(x_train)
x_testsc = sc.transform(x_test)

model2 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model2.fit(x_trainsc, y_train)
y_predict2 = model2.predict(x_testsc)

# Collect user input
user_input = {
    "brand_name": st.text_input("Enter the brand name: "),
    "os": st.text_input("Enter the os type: "),
    "screen_size": st.number_input("Enter the screen size: "),
    "4g": st.text_input("Enter yes/no for 4G: "),
    "5g": st.text_input("Enter yes/no for 5G: "),
    "main_camera_mp": st.number_input("Enter the main camera MP: "),
    "selfie_camera_mp": st.number_input("Enter the selfie camera MP: "),
    "int_memory": st.number_input("Enter the internal memory: "),
    "ram": st.number_input("Enter the RAM: "),
    "battery": st.number_input("Enter the battery capacity: "),
    "weight": st.number_input("Enter the weight: "),
    "release_year": st.number_input("Enter the release year: "),
    "days_used": st.number_input("Enter the number of days used: "),
    "normalized_new_price": st.number_input("Enter the normalized new price: "),
}

if user_input is not None:
# Convert input to DataFrame
    x = pd.DataFrame([user_input])

# Assuming 'model2' is the trained model
# Ensure consistent feature names and order
    x = pd.get_dummies(x, columns=["brand_name", "os", "4g", "5g"])

#This must be done inorder to avoid the error of mising columns caused by get_dummies
    x = x.reindex(columns=X.columns, fill_value=0)

# Scale the input data
    x = sc.transform(x)

# Make predictions
    y_predict = model2.predict(x)
    
    st.write(f"Your used item can go for: {y_predict}")



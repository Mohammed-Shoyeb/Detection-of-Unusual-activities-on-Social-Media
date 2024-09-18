# Import necessary libraries
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import json
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle as cpickle
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from datetime import datetime
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize main application window
main = tkinter.Tk()
main.title("Spammer Detection")  # Set the title of the main window
main.geometry("1300x1200")  # Set the size of the window

# Global variables for file handling and model performance metrics
global filename
global classifier
global cvv
global total, fake_acc, spam_acc
global eml_acc, random_acc, nb_acc, svm_acc
global X_train, X_test, y_train, y_test

# Function to process text by removing punctuation and stop words
def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)  # Join characters back into a string
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

# Function to upload a Twitter profile dataset
def upload():  
    global filename
    filename = filedialog.askdirectory(initialdir=".")  # Open a file dialog to select a directory
    pathlabel.config(text=filename)  # Update the label with the selected directory path
    text.delete('1.0', END)  # Clear the text area
    text.insert(END, filename + " loaded\n")  # Display a message indicating the dataset is loaded

# Function to load the Naive Bayes classifier and feature vectorizer
def naiveBayes():
    global classifier
    global cvv
    text.delete('1.0', END)  # Clear the text area
    classifier = cpickle.load(open('model/naiveBayes.pkl', 'rb'))  # Load the Naive Bayes model
    cv = CountVectorizer(decode_error="replace", vocabulary=cpickle.load(open("model/feature.pkl", "rb")))  # Load the feature vectorizer
    cvv = CountVectorizer(vocabulary=cv.get_feature_names(), stop_words="english", lowercase=True)  # Create a CountVectorizer instance
    text.insert(END, "Naive Bayes Classifier loaded\n")  # Display a message indicating the classifier is loaded

# Function to detect fake accounts and spam content from tweets
def fakeDetection():
    global total, fake_acc, spam_acc
    total = 0  # Initialize total counter
    fake_acc = 0  # Initialize fake account counter
    spam_acc = 0  # Initialize spam content counter
    favourite = '0'
    text.delete('1.0', END)  # Clear the text area
    dataset = 'Favourites,Retweets,Following,Followers,Reputation,Hashtag,Fake,class\n'  # Header for feature dataset
    for root, dirs, files in os.walk(filename):  # Walk through the directory to find JSON files
        for fdata in files:
            with open(root + "/" + fdata, "r") as file:  # Open each JSON file
                total += 1  # Increment total count
                data = json.load(file)  # Load JSON data
                textdata = data['text'].strip('\n')  # Get the tweet text
                textdata = textdata.replace("\n", " ")  # Replace newlines with spaces
                textdata = re.sub('\W+', ' ', textdata)  # Remove non-word characters
                retweet = data['retweet_count']  # Get retweet count
                followers = data['user']['followers_count']  # Get followers count
                density = data['user']['listed_count']  # Get user reputation
                following = data['user']['friends_count']  # Get following count
                replies = data['user']['favourites_count']  # Get number of replies
                hashtag = data['user']['statuses_count']  # Get hashtag count
                username = data['user']['screen_name']  # Get username
                urls_count = data['user']['utc_offset']  # Get URL count (UTC offset)

                # Handle cases where URL count might be None
                if urls_count is None:
                    urls_count = 0
                else:
                    urls_count = str(abs(int(urls_count)))  # Convert to absolute integer

                # Extract account creation date
                create_date = data['user']['created_at']
                strMnth = create_date[4:7]  # Get month as string
                day = create_date[8:10]  # Get day
                year = create_date[26:30]  # Get year

                # Convert month name to month number
                month_map = {
                    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                }
                strMnth = month_map.get(strMnth, strMnth)  # Map month name to number
                create_date = day + "/" + strMnth + "/" + year  # Format date
                create_date = datetime.strptime(create_date, '%d/%m/%Y')  # Convert to datetime object
                today = datetime.today()  # Get today's date
                age = today - create_date  # Calculate account age
                words = textdata.split(" ")  # Split tweet text into words

                # Display extracted data in the text area
                text.insert(END, "Username : " + username + "\n")
                text.insert(END, "Tweet Text : " + textdata + "\n")
                text.insert(END, "Retweet Count : " + str(retweet) + "\n")
                text.insert(END, "Following : " + str(following) + "\n")
                text.insert(END, "Followers : " + str(followers) + "\n")
                text.insert(END, "Reputation : " + str(density) + "\n")
                text.insert(END, "Hashtag : " + str(hashtag) + "\n")
                text.insert(END, "Num Replies : " + str(replies) + "\n")
                text.insert(END, "Favourite Count : " + str(favourite) + "\n")
                text.insert(END, "Created Date : " + str(create_date) + " & Account Age : " + str(age) + "\n")
                text.insert(END, "URL's Count : " + str(urls_count) + "\n")
                text.insert(END, "Tweet Words Length : " + str(len(words)) + "\n")

                # Perform spam detection using the loaded model
                test = cvv.fit_transform([textdata])  # Transform text data into feature vectors
                spam = classifier.predict(test)  # Predict if tweet is spam
                cname = 0
                fake = 0
                if spam == 0:  # If not spam
                    text.insert(END, "Tweet text contains : Non-Spam Words\n")
                    cname = 0
                else:  # If spam
                    spam_acc += 1
                    text.insert(END, "Tweet text contains : Spam Words\n")
                    cname = 1

                # Determine if account is fake based on follower and following counts
                if followers < following:
                    text.insert(END, "Twitter Account is Fake\n")
                    fake = 1
                    fake_acc += 1
                else:
                    text.insert(END, "Twitter Account is Genuine\n")
                    fake = 0

                text.insert(END, "\n")
                # Prepare the dataset for storage
                value = str(replies) + "," + str(retweet) + "," + str(following) + "," + str(followers) + "," + str(density) + "," + str(hashtag) + "," + str(fake) + "," + str(cname) + "\n"
                dataset += value

    # Save the features to a text file
    with open("features.txt", "w") as f:
        f.write(dataset)

# Function to make predictions based on the test set
def prediction(X_test, cls):
    y_pred = cls.predict(X_test)  # Predict outcomes using the classifier
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))  # Print each test instance with its prediction
    return y_pred

# Function to calculate and display accuracy
def cal_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    precision = precision_score(y_test, y_pred)  # Calculate precision
    recall = recall_score(y_test, y_pred)  # Calculate recall
    f1 = f1_score(y_test, y_pred)  # Calculate F1 score
    return accuracy, precision, recall, f1  # Return all metrics

# Function to train and test the classifiers
def train_test(cls, model_name):
    global X_train, X_test, y_train, y_test
    df = pd.read_csv('features.txt')  # Load the features from the text file
    X = df.drop(['class'], axis=1)  # Features
    y = df['class']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Split data into training and testing sets
    cls.fit(X_train, y_train)  # Train the model
    y_pred = prediction(X_test, cls)  # Make predictions
    accuracy, precision, recall, f1 = cal_accuracy(y_test, y_pred)  # Calculate metrics
    text.insert(END, f"{model_name} Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n")  # Display results

# Function to train and test a Random Forest classifier
def randomForest():
    cls = RandomForestClassifier(n_estimators=100)  # Create a Random Forest classifier
    train_test(cls, "Random Forest")  # Train and test the model

# Function to train and test a Naive Bayes classifier
def naiveBayes():
    cls = MultinomialNB()  # Create a Naive Bayes classifier
    train_test(cls, "Naive Bayes")  # Train and test the model

# Function to train and test a Support Vector Machine classifier
def supportVectorMachine():
    cls = svm.SVC()  # Create a Support Vector Machine classifier
    train_test(cls, "Support Vector Machine")  # Train and test the model

# Function to train and test an Extreme Learning Machine classifier
def extremeLearningMachine():
    cls = GenELMClassifier(n_hidden=1000, activation='sigmoid')  # Create an Extreme Learning Machine classifier
    train_test(cls, "Extreme Learning Machine")  # Train and test the model

# GUI setup: Create and place GUI elements
upload_button = Button(main, text="Upload Dataset", command=upload)  # Button to upload dataset
upload_button.pack()  # Add button to the window

pathlabel = Label(main, text="No Dataset Loaded")  # Label to display loaded dataset
pathlabel.pack()  # Add label to the window

text = Text(main, height=30, width=150)  # Text area to display output
text.pack()  # Add text area to the window

# Buttons for loading classifiers and running spam detection
naive_button = Button(main, text="Load Naive Bayes Classifier", command=naiveBayes)
naive_button.pack()

detect_button = Button(main, text="Detect Fake Accounts & Spam", command=fakeDetection)
detect_button.pack()

# Buttons to train and test various classifiers
rf_button = Button(main, text="Train Random Forest", command=randomForest)
rf_button.pack()

nb_button = Button(main, text="Train Naive Bayes", command=naiveBayes)
nb_button.pack()

svm_button = Button(main, text="Train Support Vector Machine", command=supportVectorMachine)
svm_button.pack()

elm_button = Button(main, text="Train Extreme Learning Machine", command=extremeLearningMachine)
elm_button.pack()

# Run the main application loop
main.mainloop()

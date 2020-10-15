# A Disaster Response Pipeline Project

## Motivation

In times of natural disaster, aid services are often overwhelmed with requests for assistance. Classifying these natural language requests into categories and assigning them to the correct teams becomes vital.
Using data from Figure Eight Inc, I have built a web appp that analyses such incoming messages and classifies them into the correct categories
  
## Files Included:
1.) The Data folder : A jupyter notebook file with the code  
2.) The Model folder : The dataset used for the analysis. Please unzip before using. All rights reserved with Udacity.  
3.) The App folder : Used for storing feature transformed datasets for easier modeling. Not required - generated in the code.  

## Instructions:
Run the following commands in the project's root directory to set up your database and model :
1. To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
2. To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
3. Run the following command in the app's directory to run your web app: python run.py
4. Go to http://0.0.0.0:3001/


## Prerequisites  
### Languages :  
Python 3   

### Libraries
Pandas
nltk
Sqlalchemy 
Sklearn
Pickle
Plotly
Flask
JSON

### Optional  
Jupyter Notebook  

## Summary of the Results :

Features Chosen :  
1.) **Categorical Features**: Gender, Level (free vs paid), Browser and Platform (OS). State and City were initially considered but there were too many categories if we considered them, so they were dropped.  

2.) **Numerical features** : Page events (Add friend, Add to Playlist etc), No. of songs, No. of artists, No. of sessions, listening time, age (how long the person has been a user for).   

Two approaches were chosen :   
1.) **User Based**: Modeling the differences between users based on their overall behavior  
2.) **Event Based**: Time sensitive, modeling the features that led up to the churn event.   
  
Two Models were used :   
1.) **Logistic Regression**  
2.) **Random Forest (& Gradient Boosted Trees)**  
  
  
In the end, the second approach turned out to have more accurate results, with the Logistic regression model the better option.  
![Results Image](https://github.com/adityanawal/Sparkify/blob/main/results.JPG)

## Authors
Aditya Nawalgaria

## Acknowledgments
The team of the [Udacity Nanodegree program](www.udacity.com)  

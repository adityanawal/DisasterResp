# A Disaster Response Pipeline Project

## Motivation

In times of natural disaster, aid services are often overwhelmed with requests for assistance. Classifying these natural language requests into categories and assigning them to the correct teams becomes vital.
Using data from Figure Eight Inc, I have built a web appp that analyses such incoming messages and classifies them into the correct categories
  
## Files Included:
  
1. The data folder :  
    disaster_categories.csv: A CSV dataset with the relevant disaster management categories  
    disaster_messages.csv: A CSV dataset with all the aid messages  
    process_data.py: A script with the ETL pipelineto read, clean, and save data into an SQLite database  
    DisasterResponse.db: A SQlite database with the cleaned and merged data from the ETL pipeline  
      
2. The models folder:  
    train_classifier.py: A script with the ML pipeline to train the classifier and export the model  
    classifier.pkl: The trained ML model for classifying future messages  
      
3. The app folder :   
    run.py: A script to run the web application  
    templates: Contains the html for the web application  
  
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

1. An ETL pipeline was made which would read data from csv files, clean it with relevant assumptions, merge the data and store it into an sqlite database
2. A machine learning pipepline was made that used TFIDF and Random Forest for trainign a classifier. This was further optimized by a gridsearch and the output classifier was evaluated for precision. The final classifier was stored in a pickle file.
3. A Web app was created using Flask to classify any message a user may enter into the relevant categories, using the classifier built in the previous step


## Authors
Aditya Nawalgaria

## Acknowledgments
The team of the [Udacity Nanodegree program](www.udacity.com)  
[Figure Eight](https://appen.com/) for the data  

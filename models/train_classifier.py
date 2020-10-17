import sys
import pandas as pd
import nltk
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    '''
    Load cleaned data from database
    
    INPUT:
    database_filepath : Path to the database
    OUTPUT:
    X : The messages column / feature variable
    y : The features columns / target variables
    category_names: a list of the categories
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse',engine)
    
    #set the feature variable X and target variable y
    X =  df['message']
    y = df.iloc[:,4:]
    
    #pull in the category names
    category_names = y.columns
    
    return X, y, category_names
    

def tokenize(text):
    '''
    Tokenizing text data for use with ML models
    
    INPUT:
    text : the text to be tokenized
    OUTPUT:
    clean_tokens : the tokens for the said text
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    ML model which takes in messages and classifies them on categories 
    
    INPUT:
    none
    OUTPUT:
    cv : the ML model
    '''

    
    #split data
    #X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    #first pipeline - replaced with new version
    '''
    pipeline_v1 = Pipeline([
        ('tf_vect', TfidfVectorizer(tokenizer = tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    # train data on first pipeline
    pipeline_v1.fit(X_train, y_train)
    
    #Fit and check prediction report
    y_pred = pipeline_v1.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=y.columns.values))
   
    #Gridsearch
    
    parameters = {
            #TFIDF Parameters 
            'tf_vect__max_df': (0.8, 1.0),
    
            #Random Forest Parameters
            'clf__estimator__n_estimators': [50, 100]
            'clf__estimator__min_samples_split': [2, 4]
            }

    cv = GridSearchCV(pipeline_v1, param_grid=parameters, n_jobs=-1)
    cv.fit(X_train, y_train)
    y_pred = cv.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=y.columns.values))
    '''
    
    #Second pipeline (better performance)
    pipeline_v2 = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('cvt', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    #pipeline_v2.fit(X_train, y_train)
    #y_pred = pipeline_v2.predict(X_test)
    #print(classification_report(y_test, y_pred, target_names=y.columns.values))
    
    #Gridsearch pipeline 2
    parameters = {
            #Text Pipeline - Countvectorizer Parameters 
            #'features__text_pipeline__cvt__min_df': [1, 5],
            
            #Text Pipeline - Tfidf Parameters
            'features__text_pipeline__tfidf__use_idf': (True, False),
            
            #Adaboost Forest Parameters
            #'clf__estimator__min_samples_split': [2, 4]}
            'clf__estimator__n_estimators': [50, 100]
            }
    
    cv = GridSearchCV(pipeline_v2, param_grid=parameters, n_jobs=-1, verbose =5)
    #cv.fit(X_train, y_train)
    #y_pred = cv.predict(X_test)
    #print(classification_report(y_test, y_pred, target_names=y.columns.values))     
    
    return cv
    
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance using the precision, recall and f1 support metrics
    
    INPUT:
    model:
    X_test: messages from the test set
    Y_test: categories from the test set
    category_names: name of the categories
    
    OUTPUT:
    none
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    '''
    Save model to a path
    
    INPUT:
    model: model to be saved
    model_filepath : Path for saving the model
    
    OUTPUT:
    none
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
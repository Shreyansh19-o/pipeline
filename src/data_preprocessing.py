import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

#ensure the logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

#setting up logger configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """
    Transform the text by removing punctuation, converting to lowercase,
    removing stopwords, and stemming.
    """
    ps=PorterStemmer()
        #convert to lowercase
    text=text.lower()
        #tokenize the text
    text=nltk.word_tokenize(text)
        #remove non alphabetic tokens
    text=[word for word in text if word.isalpha()]
        #Remove stopwords and punctuation
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
        #stem the words
    text=[ps.stem(word)for word in text]
        # Join the tokens back into a single string
    return " ".join(text)

def preprocess_data(df, text_column='text', target_column='target'):
    """
    Preprocess the data by removing duplicates and handling missing values.
    """
    try:
        logger.debug('Starting data preprocessing')
        #encode the target column
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')
        
        #remove duplicates rows
        df=df.drop_duplicates( keep='first')
        logger.debug('Duplicates removed')
        
        #Apply the text transformation function to the text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text transformation applied')
        return df
    
    except KeyError as e:
        logger.error('Missing columns in the DataFrame: %s', e)
        raise
    except Exception as e:
        logger.error('Error during normalization: %s', e)
        raise
    
def main(text_column='text', target_column='target'):
    """
    Main function to run the preprocessing steps.
    """
    
    try:
        # fetch the data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded from CSV files')
        
        
        #Transform the text data
        train_processed_data = preprocess_data(train_data, text_column, target_column)
        test_processed_data = preprocess_data(test_data, text_column, target_column)
        
        
        #store the processed data
        data_path=os.path.join("./data","interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)    
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
        
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('Empty DataFrame: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
        
        
    
       
    
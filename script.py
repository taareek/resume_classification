import pandas as pd
import numpy as np
import re
import nltk
import PyPDF2
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import xgboost as xgb
import os
from os import walk
import shutil
import csv

# parsing directory path as variable
import argparse
parser = argparse.ArgumentParser(description='Source directory path')
parser.add_argument('dir_path', type=str)
args = parser.parse_args()


# initializing NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# getting multiclass feature selector pickle file
tfid_path = "models/fe_xgb_tfid"
feats_in = open(tfid_path, "rb")
tfid_ft_extractor = pickle.load(feats_in)

# loading feature extractor
fs_path = "models/ft_selection_xgb_tfid"
fs_in = open(fs_path, "rb")
fs_selector = pickle.load(fs_in)

# loading model
model_path = "models/XGB_final"
model_in = open(model_path, 'rb')
loaded_model = pickle.load(model_in)


# function to get files within a sub-directory
def get_subdir_files(src_dir):
  sub_files = []
  only_names = []
  for file in os.listdir(src_dir):
    if(os.path.isfile(os.path.join(src_dir, file))):
      file_name = os.path.join(src_dir, file)
      sub_files.append(file_name)
      only_names.append(file)
  return sub_files, only_names

# function to get all files within subdirectoris as well
def get_all_files_with_name(src_dir):
  all_files_path = []
  all_files_name = []
  for path, subdirs, files in os.walk(src_dir):
    for name in files:
      # file name with path
      file_name = os.path.join(path, name)
      all_files_path.append(file_name)
      # get the names 
      all_files_name.append(name)
  return all_files_path, all_files_name

# functon to create a directory
def make_directory(root_path, dir_name):
  # defining path
  path = os.path.join(root_path, dir_name)

  # we can give a directory name that already exist by mistake, therefore, using try except
  try:
    os.mkdir(path)
    print(f"Directory created successfully at: {path}")
  except OSError as error:
      print(error)
  return path


# text cleaning
def preprocess_text(txt):
    # convert all characters in the string to lower case
    txt = txt.lower()
    # remove non-english characters, punctuation and numbers
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = re.sub('http\S+\s*', ' ', txt)  # remove URLs
    txt = re.sub('RT|cc', ' ', txt)  # remove RT and cc
    txt = re.sub('#\S+', '', txt)  # remove hashtags
    txt = re.sub('@\S+', '  ', txt)  # remove mentions
    txt = re.sub('\s+', ' ', txt)  # remove extra whitespace
    # tokenize word
    txt = nltk.tokenize.word_tokenize(txt)
    # remove stop words
    txt = [w for w in txt if not w in nltk.corpus.stopwords.words('english')]

    return ' '.join(txt)


# function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# then processes those raw data
def process_resume(resumes):
  resume_text = []
  for i in range(len(resumes)):
    # getting text
    raw_text = extract_text_from_pdf(resumes[i])
    # preprocess those text
    processed_text = preprocess_text(raw_text)
    # append to the list
    resume_text.append(processed_text)
  return resume_text

# now we will extract features from resume text and perform feature selection 
def feat_extract_n_select(resume_text):
  resume_features = tfid_ft_extractor.transform(resume_text)
  selected_resume_feat = fs_selector.transform(resume_features)
  return selected_resume_feat


# our model prediction and categorization of resumes 
def categorize_resumes(model, resumes, resume_names, resume_paths, labels_list, input_dir):
  categories_resumes = []
  a = 0
  for resume in resumes:
    resume_name = resume_names[a]
    resume_path = resume_paths[a]
    pred_categ_label = model.predict(resume)
    pred_categ_name = labels_list[int(pred_categ_label)]

    # appending file and predicted category into a list
    categories_resumes.append([resume_path, pred_categ_name])

    # defining destination folder name, input folder based on requirements 
    dest_name= os.path.join(input_dir+'/'+pred_categ_name)
    # checking is there any folder named by the predicted class to the dest folder
    if(os.path.isdir(dest_name)):
      shutil.move(resume_path, dest_name+'/'+resume_name)
    else:
      new_dir = make_directory(input_dir+'/', pred_categ_name)
      shutil.move(resume_path, new_dir+'/'+resume_name)
    a+=1

  # making csv file to the destination folder
  with open(input_dir+'/'+ 'categorized_resumes.csv', 'w', encoding='UTF8', newline='') as p:
    writer = csv.writer(p)
    writer.writerows(categories_resumes)


## our defined category-integer mapping
labels_dict = {
    'AUTOMOBILE': 0,
    'AVIATION': 1,
    'BUSINESS-DEVELOPMENT': 2,
    'APPAREL': 3, 'ARTS': 4,
    'ACCOUNTANT': 5,
    'BPO': 6,
    'BANKING': 7,
    'ADVOCATE': 8,
    'AGRICULTURE': 9,
    'CONSTRUCTION': 10,
    'FITNESS': 11,
    'CONSULTANT': 12,
    'DIGITAL-MEDIA': 13,
    'ENGINEERING': 14,
    'DESIGNER': 15,
    'FINANCE': 16,
    'CHEF': 17,
    'HR': 18,
    'HEALTHCARE': 19,
    'PUBLIC-RELATIONS': 20,
    'INFORMATION-TECHNOLOGY': 21,
    'SALES': 22,
    'TEACHER': 23
    }

# to get as our predicted class as labels 
text_labels = list(labels_dict.keys())





def main():   
    # defining source directory
    input_dir = args.dir_path
    # read all the files from the given directory, and their names
    all_test_resumes, all_resume_names = get_all_files_with_name(input_dir)

    # preprocess resume data
    processed_resume_text = process_resume(all_test_resumes)

    # apply feature extraction and selection
    X_test = feat_extract_n_select(processed_resume_text)

    # perform resume categarization
    categorize_resumes(loaded_model, X_test, all_resume_names, all_test_resumes, text_labels, input_dir)

if __name__ == "__main__":
    main()

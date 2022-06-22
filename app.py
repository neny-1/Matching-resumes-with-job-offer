#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 01:56:03 2022

@author: mohamed
"""


#**************** start imports ****************#
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics

from sklearn.datasets import load_diabetes, load_boston
import textract
import PyPDF2

import string 
from wordcloud import WordCloud 

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
english_stopwords = stopwords.words('english')

import en_core_web_sm
from spacy.pipeline import EntityRuler
from spacy import displacy
import jsonlines
import os

from pylab import rcParams
sns.set(style='whitegrid', palette='muted', font_scale=.4)
rcParams['figure.figsize'] = 35, 12

#**************** end imports ****************#


#=============== start model one function ===============#
def model_one(model_cv):
    
    #read csv file
    resumeDataSet = pd.read_csv(model_cv)
    resumeDataSet['cleaned_resume'] = ''
    st.write(resumeDataSet.head())
    
    #display categories and number of resume
    st.title("Displaying the distinct categories of resume and the number of records belonging to each category:\n\n")
    st.table(resumeDataSet['Category'].value_counts())
    
    #by using seaborn plotting number of categories  
    st.title("The most efficient and majority categories exist in dataset ")    
    plt.figure(figsize=(20,5))
    plt.xticks(rotation=90)
    ax=sns.countplot(x="Category", data=resumeDataSet)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
    plt.grid()
    st.pyplot(fig= plt.grid())

    
    #by using matplolip show percentege of categories
    st.title("The most efficient and majority categories exist by percentage ")
    from matplotlib.gridspec import GridSpec   
    targetCounts = resumeDataSet['Category'].value_counts()
    targetLabels  = resumeDataSet['Category'].unique()
    
    # Make square figures and axes
    plt.figure(1, figsize=(22,22))
    the_grid = GridSpec(2, 2)

    cmap = plt.get_cmap('coolwarm') 
    plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')
    
    source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True)
    st.pyplot(fig=plt.show())

    
    #clean text 
    st.title("")
    import re
    def cleanResume(resumeText):
        resumeText = re.sub('http\S+\s*', ' ', resumeText)
        resumeText = re.sub('RT|cc', ' ', resumeText)  
        resumeText = re.sub('#\S+', '', resumeText)  
        resumeText = re.sub('@\S+', '  ', resumeText)  
        resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText) 
        resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
        resumeText = re.sub('\s+', ' ', resumeText) 
        return resumeText
    st.write("in uploaded resumes after passing it to cleaning process:")
    resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))    
    cleaned_resume = resumeDataSet['cleaned_resume']
    st.write(cleaned_resume)
    resumeDataSet_d=resumeDataSet.copy()
    
    
    #stop words (and/or/is/the/are)
    one_Set_Of_StopWords = set(stopwords.words('english')+['``',"''"]) 
    totalWords =[] 
    Sentences = resumeDataSet['Resume'].values 
    cleanedSentences = "" 
    for records in Sentences: 
        Cleaned_Text = cleanResume(records) 
        cleanedSentences +=  Cleaned_Text 
        requiredWords = nltk.word_tokenize(Cleaned_Text)  
        for word in requiredWords:
            if word not in one_Set_Of_StopWords and word not in string.punctuation:
                totalWords.append(word) 
    Word_freq_dist = nltk.FreqDist(totalWords) 
    mostcommon = Word_freq_dist.most_common(4000) 
    
    #visualize most important/frequency word
    st.title("visualize most important/frequency word in uploaded file")
    wc = WordCloud().generate(cleanedSentences)
    plt.figure(figsize=(10,10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot(fig=plt.show())
    
    
    #TFIDF vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer 
    from scipy.sparse import hstack 
    requiredText = resumeDataSet['cleaned_resume']
    requiredTarget = resumeDataSet['Category'] 
    
    word_vectorizer = TfidfVectorizer(sublinear_tf=True,stop_words='english') 
    word_vectorizer.fit(requiredText)
    tfidf = word_vectorizer.transform(requiredText)
    
    #Features
    resumeDataSet['cleaned_resume'].unique()
    resumeDataSet['cleaned_resume'].unique()

    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(tfidf,requiredTarget,random_state=0, test_size=0.25,shuffle =True)
    
    
    #KNN Model 
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train, y_train)
    knn_prediction = model.predict(X_test)
    
    #st.write(knn_prediction)
    st.header("Applay KNN Algorithm on Dataset")
    st.write('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
    st.write('Accuracy of KNeighbors Classifier on test set:     {:.2f}'.format(model.score(X_test, y_test)))
   
    from sklearn.metrics import confusion_matrix as cm
    conf_mat = cm(y_test,knn_prediction)
    sns.heatmap(conf_mat, annot=True)
    
    #from import metrices
    st.write("\n Classification report for classifier %s:\n%s\n" % (model, metrics.classification_report(y_test,knn_prediction)))

################################ end model_one function ###################################


#=============== start model two for specific cv function ===============#
def model_two(file_cv):
    
    def extract_text_from_pdf(file):
        fileReader = PyPDF2.PdfFileReader(open(file,'rb'))
        page_count = fileReader.getNumPages()
        text = [fileReader.getPage(i).extractText() for i in range(page_count)]
        return str(text).replace("\\n", "")
        
    
    def extract_text_from_word(filepath):
        
        txt = textract.process(filepath).decode('utf-8')
        
        return txt.replace('\n', ' ').replace('\t', ' ')
    
    # Load pre-trained english  language model
    nlp = en_core_web_sm.load()
    
    # File Extension. set as 'pdf' or as 'doc(x)'
    extension = 'docx'
    
    def create_tokenized_texts_list(extension):

        resume_texts, resume_names = [], []
        
        # Loop over the contents of the directory containing the resumes, filtering by .pdf or .doc(x)
        resume = file_cv.name
        # Read in every resume with .doc or .docx extension in the directory
        resume_texts.append(nlp(extract_text_from_word(r"model_2/dataset/" + resume)))
        resume_names.append(resume.split('_')[0].capitalize())
    
        return resume_texts, resume_names
    
    with jsonlines.open("model_2/skill_patterns.jsonl") as f:
        created_entities = [line['label'].upper() for line in f.iter()]
    
    def add_newruler_to_pipeline(skill_pattern_path):
        
        new_ruler = nlp.add_pipe("entity_ruler", after='parser')
        new_ruler.from_disk(skill_pattern_path)
        
    def create_skill_set(doc):
    
        return set([ent.label_.upper()[6:] for ent in doc.ents if 'skill' in ent.label_.lower()]) 
    def create_skillset_dict(resume_names, resume_texts):
    
        skillsets = [create_skill_set(resume_text) for resume_text in resume_texts]
    
        return dict(zip(resume_names, skillsets))


    def match_skills(vacature_set, cv_set, resume_name):
        
        
        if len(vacature_set) < 1:
            print('could not extract skills from job offer text')
        else:
            pct_match = round(len(vacature_set.intersection(cv_set[resume_name])) / len(vacature_set) * 100, 0)
            print(resume_name + " has a {}% skill match on this job offer".format(pct_match))
            print('Required skills: {} '.format(vacature_set))
            print('Matched skills: {} \n'.format(vacature_set.intersection(skillset_dict[resume_name])))
    
            return (resume_name, pct_match)
    
    
    add_newruler_to_pipeline(r"model_2/skill_patterns.jsonl")
    
    
    resume_texts, resume_names = create_tokenized_texts_list(extension)
        
    # read job offer 
    pdfFileObject = open(r"model_2/job offer .pdf", 'rb')
    
    pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
    
    # No. Of Pages :  pdfReader.numPages)
    pageObject = pdfReader.getPage(0)

    vacature_text = pageObject.extractText()
    
    # example of job offer text (string). Can input your own.
    vacature_text ="Founded in 2018, MaxAB is a rapidly growing food and grocery B2B e-commerce and distribution platform that serves a network of traditional retailers (mom-and-pop stores) across the MENA region. Using proprietary technology, MaxAB offers a transformative pull-driven supply chain and a tech-product that empowers both traditional retailers and suppliers.MaxAB offers traditional retailers the simplicity of dealing with one supplier, transparent pricing, on-demand delivery, and a range of value-added and embedded finance solutions. Suppliers benefit from MaxAB’s end-to-end supply chain solutions and business intelligence tools that allow them to accurately predict, monitor, and control the impact of their strategies in real time.Our MaxAB talent are dedicated to upholding the MaxAB culture and values all while continuing to grow and improve services for our clients. They are innovating new ways to help improve the quality of life of the Egyptian retailer and soon to other retailers globally.If you are passionate about working hard to make an impact and innovate new solutions, MaxAB is looking for top talent.Job Summary:The Senior Data Scientist applies and inspires the adoption of advanced data science and analytics across the business.Responsibilities:• Gather requirements from business teams and translate them into problem statements• Solve complex business problems with data and use data science tools to find solutions for challenges in the realm of both big & small data• Mine and analyze data from company databases to drive optimization and improvement of product development, marketing techniques, and business strategies• Develop custom data models and algorithms to apply to data sets• Develop processes and tools to monitor and analyze model performance and data accuracy" 
    
    # Create a set of the skills extracted from the job offer text
    vacature_skillset = create_skill_set(nlp(vacature_text))
    
    # Create a list with tuple pairs containing the names of the candidates and their match percentage
    skillset_dict = create_skillset_dict(resume_names, resume_texts)
    match_pairs = [match_skills(vacature_skillset, skillset_dict, name) for name in skillset_dict.keys()]


    match_pairs.sort(key=lambda tup: tup[1], reverse=True)
    
    # Unpack tuples
    names, pct = zip(*match_pairs)
    
    # Plotting
    sns.set_theme(style='darkgrid')
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_title('Job offer match with CVs', fontsize=20)
    ax.set(xlabel='CVs', ylabel='% Match')
    ax.set(ylim=(0, 100))
    
    sns.set(font_scale=1.5)
    sns.barplot(x=list(names), y=list(pct), color='b')
    
    st.pyplot(plt.show())

################################ end model_two for specific cv function ###################################


#========= start model two for all cvs in dataset folder =========#
def model_two_all():

    
    def extract_text_from_pdf(file):
    
        fileReader = PyPDF2.PdfFileReader(open(file,'rb'))
        page_count = fileReader.getNumPages()
        text = [fileReader.getPage(i).extractText() for i in range(page_count)]
        
        return str(text).replace("\\n", "")
        
    
    def extract_text_from_word(filepath):
        
        txt = textract.process(filepath).decode('utf-8')
        
        return txt.replace('\n', ' ').replace('\t', ' ')
    import en_core_web_sm
    
    # Load pre-trained english  language model
    nlp = en_core_web_sm.load()
    
    # File Extension. set as 'pdf' or as 'doc(x)'
    extension = ''
    
    def create_tokenized_texts_list(extension):
    
        resume_texts, resume_names = [], []
        
        # Loop over the contents of the directory containing the resumes, filtering by .pdf or .doc(x)
        for resume in list(filter(lambda x: extension in x, os.listdir("model_2/dataset/"))):
            st.write(resume)
            resume_texts.append(nlp(extract_text_from_word(r"model_2/dataset/" + resume)))
            resume_names.append(resume.split('_')[0].capitalize())
            
            
        return resume_texts, resume_names

 

    with jsonlines.open("model_2/skill_patterns.jsonl") as f:
        created_entities = [line['label'].upper() for line in f.iter()]
    
    def add_newruler_to_pipeline(skill_pattern_path):
        
        new_ruler = nlp.add_pipe("entity_ruler", after='parser')
        new_ruler.from_disk(skill_pattern_path)
        
    def create_skill_set(doc):
    
        return set([ent.label_.upper()[6:] for ent in doc.ents if 'skill' in ent.label_.lower()]) 
    def create_skillset_dict(resume_names, resume_texts):
    
        skillsets = [create_skill_set(resume_text) for resume_text in resume_texts]
    
        return dict(zip(resume_names, skillsets))


    def match_skills(vacature_set, cv_set, resume_name):
        
        
        if len(vacature_set) < 1:
            print('could not extract skills from job offer text')
        else:
            pct_match = round(len(vacature_set.intersection(cv_set[resume_name])) / len(vacature_set) * 100, 0)
            print(resume_name + " has a {}% skill match on this job offer".format(pct_match))
            print('Required skills: {} '.format(vacature_set))
            print('Matched skills: {} \n'.format(vacature_set.intersection(skillset_dict[resume_name])))
    
            return (resume_name, pct_match)
    
    
    add_newruler_to_pipeline(r"model_2/skill_patterns.jsonl")
    
    
    resume_texts, resume_names = create_tokenized_texts_list(extension)
        
    # read job offer 
    pdfFileObject = open(r"model_2/job offer .pdf", 'rb')
    
    pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
    
    # No. Of Pages :  pdfReader.numPages)
    pageObject = pdfReader.getPage(0)
    
    vacature_text = pageObject.extractText()
    
    
    # example of job offer text (string). Can input your own.
    vacature_text ="Founded in 2018, MaxAB is a rapidly growing food and grocery B2B e-commerce and distribution platform that serves a network of traditional retailers (mom-and-pop stores) across the MENA region. Using proprietary technology, MaxAB offers a transformative pull-driven supply chain and a tech-product that empowers both traditional retailers and suppliers.MaxAB offers traditional retailers the simplicity of dealing with one supplier, transparent pricing, on-demand delivery, and a range of value-added and embedded finance solutions. Suppliers benefit from MaxAB’s end-to-end supply chain solutions and business intelligence tools that allow them to accurately predict, monitor, and control the impact of their strategies in real time.Our MaxAB talent are dedicated to upholding the MaxAB culture and values all while continuing to grow and improve services for our clients. They are innovating new ways to help improve the quality of life of the Egyptian retailer and soon to other retailers globally.If you are passionate about working hard to make an impact and innovate new solutions, MaxAB is looking for top talent.Job Summary:The Senior Data Scientist applies and inspires the adoption of advanced data science and analytics across the business.Responsibilities:• Gather requirements from business teams and translate them into problem statements• Solve complex business problems with data and use data science tools to find solutions for challenges in the realm of both big & small data• Mine and analyze data from company databases to drive optimization and improvement of product development, marketing techniques, and business strategies• Develop custom data models and algorithms to apply to data sets• Develop processes and tools to monitor and analyze model performance and data accuracy" 
    
    # Create a set of the skills extracted from the job offer text
    vacature_skillset = create_skill_set(nlp(vacature_text))
    
    # Create a list with tuple pairs containing the names of the candidates and their match percentage
    skillset_dict = create_skillset_dict(resume_names, resume_texts)
    match_pairs = [match_skills(vacature_skillset, skillset_dict, name) for name in skillset_dict.keys()]
    
    match_pairs.sort(key=lambda tup: tup[1], reverse=True)
    
    # Unpack tuples
    names, pct = zip(*match_pairs)
    
    # Plotting
    sns.set_theme(style='darkgrid')
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_title('Job offer match with CVs', fontsize=20)
    ax.set(xlabel='CVs', ylabel='% Match')
    ax.set(ylim=(0, 100))
    
    
    sns.set(font_scale=1.5)
    sns.barplot(x=list(names), y=list(pct), color='b')
    
    st.pyplot(plt.show())
    
################################ end model two for all cvs in dataset folder function ######################################


#========= start Streamlit App =========#
#menu
menu =["home", "Login", "sign up"]
choice = st.sidebar.selectbox("menu", menu)

# home page
if choice == "home":
    st.subheader("Hello in Smart Recuritment System")
    st.image("images/1.jpeg")

#login page
elif choice == "Login":
    st.subheader("Welcome in Smart HR Application")


    username = st.sidebar.text_input("user name")
    password = st.sidebar.text_input("password",type='password')
    
    if st.sidebar.checkbox("Login"):
        if password =='1' and username =='neny':                

           st.success("Logged in as {}".format(username))
           st.image("images/6.jpeg")

           ChooseModel = st.selectbox("Choose Model",["Welcome","model1","model2"])
           if ChooseModel =="Welcome":
               st.title("Hello in HR page")
          
           # model one 
           elif ChooseModel =="model1":

              st.title("category predicated model")
              st.subheader("model decription:  this model classify CVs and output the category result")
              st.image("images/4.jpeg")
              model_cv = st.file_uploader("Upload your input CSV file", type=["csv"])
   
                         
              if model_cv is not None:
               model_one(model_cv)
             
           # model two
           elif  ChooseModel =="model2" :
              st.title("Result matching percentage model")
              st.subheader("model decription: this model parsing CVs and output the matching percentage result to apply this model")
              st.image("images/7.jpeg")
              st.title("All Cvs in dataset folder")
              model_two_all() 
              st.title("Choose specific cv from dataset folder")
              uploaded_file = st.file_uploader("Upload your input CSV file")
              if uploaded_file is not None:
                  model_two(uploaded_file)
             
################################ end streamlit App ################################
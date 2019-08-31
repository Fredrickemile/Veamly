# import libaries
import pandas as pd 
import re
import numpy as np 
import matplotlib.pyplot as plt 
from nltk.corpus import stopwords
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from plotly.offline import iplot
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True, theme='ggplot')

data=pd.read_csv('Data/github_comments.tsv',delimiter='\t',encoding='utf-8', index_col=0)
print(data.head())

def print_comment(index):
    example= data[data.index == index][['comment','comment_date']].values[0]
    if len(example)> 0:
        print(example[0])
        print('comment date:', example[1])
print_comment(20)

def get_top_n_words(corpus, n=None):
    vec= CountVectorizer().fit(corpus)
    bag_of_words= vec.transform(corpus)
    sum_words= bag_of_words.sum(axis=0)
    words_freq=[(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq= sorted(words_freq, key= lambda x: x[1], reverse= True)
    return words_freq[:n]

common_words= get_top_n_words(data['comment'],30)
df_1= pd.DataFrame(common_words, columns=['comment','count'])
df_1.groupby('comment').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', 
                                                           linecolor='black',
                                                           title='Top 30 words in comment made by Developer before removing stop words')

def get_top_n_words(corpus, n=None):
    vec= CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words= vec.transform(corpus)
    sum_words= bag_of_words.sum(axis=0)
    words_freq=[(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq= sorted(words_freq, key= lambda x: x[1], reverse= True)
    return words_freq[:n]

common_words= get_top_n_words(data['comment'],30)
df_1= pd.DataFrame(common_words, columns=['comment','count'])
df_1.groupby('comment').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', 
                                                           linecolor='black',
                                                           title='Top 30 words in comment made by Developer after removing stop words')
def get_top_n_words(corpus, n=None):
    vec= CountVectorizer(ngram_range=(2,2)).fit(corpus)
    bag_of_words= vec.transform(corpus)
    sum_words= bag_of_words.sum(axis=0)
    words_freq=[(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq= sorted(words_freq, key= lambda x: x[1], reverse= True)
    return words_freq[:n]

common_words= get_top_n_words(data['comment'],30)
df_1= pd.DataFrame(common_words, columns=['comment','count'])
df_1.groupby('comment').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', 
                                                           linecolor='black',
                                                           title='Top 30 Bigrams in comment made by Developer before removing stop words')
def get_top_n_words(corpus, n=None):
    vec= CountVectorizer(ngram_range=(2,2),stop_words='english').fit(corpus)
    bag_of_words= vec.transform(corpus)
    sum_words= bag_of_words.sum(axis=0)
    words_freq=[(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq= sorted(words_freq, key= lambda x: x[1], reverse= True)
    return words_freq[:n]

common_words= get_top_n_words(data['comment'],30)
df_1= pd.DataFrame(common_words, columns=['comment','count'])
df_1.groupby('comment').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', 
                                                           linecolor='black',
                                                           title='Top 30 Bigrams in comment made by Developer after removing stop words')
def get_top_n_words(corpus, n=None):
    vec= CountVectorizer(ngram_range=(3,3)).fit(corpus)
    bag_of_words= vec.transform(corpus)
    sum_words= bag_of_words.sum(axis=0)
    words_freq=[(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq= sorted(words_freq, key= lambda x: x[1], reverse= True)
    return words_freq[:n]

common_words= get_top_n_words(data['comment'],30)
df_1= pd.DataFrame(common_words, columns=['comment','count'])
df_1.groupby('comment').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', 
                                                           linecolor='black',
                                                           title='Top 30 Trigrams comment made by Developer before removing stop words')
def get_top_n_words(corpus, n=None):
    vec= CountVectorizer(ngram_range=(3,3),stop_words='english').fit(corpus)
    bag_of_words= vec.transform(corpus)
    sum_words= bag_of_words.sum(axis=0)
    words_freq=[(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq= sorted(words_freq, key= lambda x: x[1], reverse= True)
    return words_freq[:n]

common_words= get_top_n_words(data['comment'],30)
df_1= pd.DataFrame(common_words, columns=['comment','count'])
df_1.groupby('comment').sum()['count'].sort_values(ascending=False).iplot(kind='bar', yTitle='Count', 
                                                           linecolor='black',
                                                           title='Top 30 Trigrams in comment made by Developer after removing stop words')
data['word_count']= data['comment'].apply(lambda x: len(str(x).split()))
comment_lengths= list(data['word_count'])
print("Number of descriptions:", len(comment_lengths),
     "\nAverage word count", round(np.average(comment_lengths),0),
     "\nMinimum word count", min(comment_lengths),
     "\nMaximum word count", max(comment_lengths))

data['word_count'].iplot(kind='hist', bins=20, linecolor='black', xTitle='word count', yTitle='count',
                        title="Word Count Distrubution in Comments")

stop_words= set(stopwords.words('english'))
replace_by_space= re.compile('[/(){}\[\]\|@,;]')
replace_by_space_symbol= re.compile('[^0-9a-z #+_]')

def clean_text(text):
    text= text.lower() #lowercase text 
    text= replace_by_space.sub(' ', text)
    text= replace_by_space_symbol.sub(' ',text)
    text= ' '.join(word for word in text.split() if word not in stop_words)# removing stop words
    return text

data['comment']= data['comment'].apply(clean_text)
data.head()

data['merged_at']= pd.to_datetime(data['merged_at'], errors='coerce')
data['merged_at'].head()
data['comment_date']= pd.to_datetime(data['comment_date'], errors='coerce')

data['ResolvedTime']= data['merged_at']-data['comment_date']
data['ResolvedTime']= round(data['ResolvedTime']/np.timedelta64(1,'h'),2)
data.head()

data.fillna(0, inplace=True)
print("The Average time spent on resolving after a change was Request is:",np.average(data['ResolvedTime']),"h",
"\nMedian value of time spent on resolving after a change was Request is  ",np.median(data['ResolvedTime']),"h",
"\nThe maximum time spent on resolving a change request is",max(data['ResolvedTime']),"h",
"\nThe minimum time spent on resolving a change request is",min(data['ResolvedTime']),"h")

fdist= FreqDist(data['comment'])
wc = WordCloud(width=800, height=400, max_words=100).generate_from_frequencies(fdist)
plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")

n_components= 5
n_clusters= 5

tfidf= TfidfVectorizer(stop_words='english')
X_test= tfidf.fit_transform(data['comment'])

svd= TruncatedSVD(n_components=n_components, random_state=0)
X_2d= svd.fit_transform(X_test)

kmeans= KMeans(n_clusters= n_clusters, random_state= 0)

X_clustered= kmeans.fit_predict(X_2d) 

df_plot= pd.DataFrame(list(X_2d),list(X_clustered))
df_plot= df_plot.reset_index()
df_plot.rename(columns={'index': 'Cluster'}, inplace= True)
df_plot['Cluster']= df_plot['Cluster'].astype(int)

print(df_plot.head())
print(df_plot.groupby('Cluster').agg({'Cluster': 'count'}))


col= df_plot['Cluster'].map({0:'b', 1:'r', 2: 'g', 3:'purple', 4:'gold'})

n= 5

fig, ax= plt.subplots(n, n, sharex= True, sharey= True, figsize= (15,15))
fig.tight_layout(rect= [0.05, 0.05, 0.95, 0.95])

k= 0
for i in range(0,n):
    for j in range(0,n):
        if i!= j:
            df_plot.plot(kind = 'scatter', x= j, y= i, c= col, ax= ax[i][j], fontsize= 18)
        else:
            ax[i][j].set_xlabel(i)
            ax[i][j].set_ylabel(j)
            ax[i][j].set_frame_on(False)
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
        
plt.suptitle('2D clustering view of the first {} components'.format(n), fontsize = 20)
fig.text(0.5, 0.01, 'Component n', ha='center', fontsize = 18)
fig.text(0.01, 0.5, 'Component n', va='center', rotation='vertical', fontsize = 18)

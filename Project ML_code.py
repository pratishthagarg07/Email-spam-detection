import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

%pip install seaborn

# # Data cleaning

data=pd.read_csv('spam.csv')

data.shape

data.drop_duplicates(inplace=True)

data.shape

data.head()

data['target']=data['Category'].map({'spam':1,'ham':0})

data.head()

data['target'].value_counts()

data.isnull().sum()

data['Category'] = data['Category'].str.strip()
data['Message'] = data['Message'].str.strip()

print((data['Message'] == '').sum())

data=data[data['Message'] != '']

data['Message'] = data['Message'].str.lower()

data['Message'] = data['Message'].apply(
    lambda x: ' '.join(x.split())
)

# # EDA
#  Exploratory Data Analysis                                                                       

print(data[data['Message'].str.len() < 3])

print(data['Message'])

data['num character'] = data['Message'].apply(len)

data['num character']

data['num words']=data['Message'].apply(lambda x:len(x.split()))

data['num words']

data['num sentences']=data['Message'].apply(lambda x:len(x.split('.')))

data['num sentences']

print(data[['Message', 'num character', 'num words','num sentences']].head())

# Count how many spam and ham messages exist
counts = data['target'].value_counts()

plt.figure(figsize=(5, 5))
plt.pie(
    counts,                           
    labels=['Ham', 'Spam'],           
    autopct='%1.1f%%',               
   colors=['darkgreen', 'lightgreen'],  
    startangle=90                     
)
plt.title('Spam vs Ham distribution')
plt.show

# Character analysis

ham  = data[data['target'] == 0]
spam = data[data['target'] == 1]

plt.figure(figsize=(10, 4))

plt.hist(ham['num character'],
         bins=50, color='Red', label='Ham',  alpha=0.7)

plt.hist(spam['num character'],
         bins=50, color='Yellow', label='Spam', alpha=0.7)

plt.xlabel('Number of characters')
plt.ylabel('Count')
plt.title('Message length — Ham vs Spam')
plt.legend()
plt.show()

# Words analysis

plt.figure(figsize=(10, 4))

plt.hist(ham['num words'],
         bins=35, color='Purple', label='Ham',  alpha=0.7)

plt.hist(spam['num words'],
         bins=35, color='Pink', label='Spam', alpha=0.7)

plt.xlabel('Number of words')
plt.ylabel('Count')
plt.title('Word count — Ham vs Spam')
plt.legend()
plt.show()

# sentences analysis
plt.figure(figsize=(10, 4))

plt.hist(ham['num sentences'],
         bins=35, color='Black', label='Ham',  alpha=0.7)

plt.hist(spam['num sentences'],
         bins=35, color='White', label='Spam', alpha=0.7)

plt.xlabel('Number of words')
plt.ylabel('Count')
plt.title('Word count — Ham vs Spam')
plt.legend()
plt.show()

features = ['num character', 'num words', 'num sentences']
correlations = [data[f].corr(data['target']) for f in features]

plt.figure(figsize=(7, 4))

bars = plt.bar(
    features,
    correlations,
    color=['violet', 'purple', 'pink']
)

for bar, val in zip(bars, correlations):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.005,
        f'{val:.3f}',
        ha='center',
        fontsize=11
    )

plt.title('Correlation of features with spam label')
plt.ylabel('Correlation')

max_corr = max(correlations)
plt.ylim(0, max_corr + 0.05)

plt.show()
#Correlation measures how strongly a feature is related to the target variable; a positive value means the feature increases
#with the target, a negative value means it decreases, and a value close to zero indicates little or no relationship.

from collections import Counter

counter = Counter()

for msg in data['Message']:
    counter.update(msg.lower().split())

top_words = counter.most_common(20)

stopwords = {word for word, count in top_words}

print(stopwords)

def get_top_words(messages, n=10):
    counter = Counter()
    for msg in messages:
        words = re.findall(r'[a-zA-Z]+', msg.lower())
        counter.update([w for w in words
                         if w not in stopwords and len(w) > 2])
    return counter.most_common(n)

spam_words = get_top_words(spam['Message'])
ham_words  = get_top_words(ham['Message'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Spam chart
ax1.barh([w[0] for w in spam_words],
          [w[1] for w in spam_words], color='blue')
ax1.invert_yaxis()   # highest bar at top
ax1.set_title('Top words in Spam')

# Ham chart
ax2.barh([w[0] for w in ham_words],
          [w[1] for w in ham_words], color='grey')
ax2.invert_yaxis()
ax2.set_title('Top words in Ham')

plt.tight_layout()   # prevents charts from overlapping
plt.show()

# Descriptive statistics — compare spam vs ham across all 3 features
data.groupby('target')[['num character', 'num words', 'num sentences']].mean().round(2)

# Spam messages tend to be longer and contain more words than ham messages,
# making message length a potentially useful feature for classification. 1=spam,0=ham

# checking

print(data.isnull().sum())
print(data.duplicated().sum())

print(data['target'].value_counts())
print(data[['Category','target']].head())

print(data[['num character','num words','num sentences']].head())

print(data[['num character','num words','num sentences']].describe())

data['target'].value_counts(normalize=True) * 100

data.groupby('target')[['num character','num words','num sentences']].mean()

print(data.columns.tolist())

data[['num character', 'num_character', 'num_characters']].head()

data = data.drop(columns=['num character', 'num_character'])

data = data.rename(columns={
    'num charatcer': 'num character'
})

# # Text Preprocessing

data.head()



import string

# Hardcoded stopwords - no download needed
stop_words = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','will','would','shall','should','may','might','must','can',
    'could','not','and','but','or','nor','so','yet','both','either','neither',
    'the','a','an','in','on','at','to','for','of','with','by','from','up',
    'about','into','through','during','before','after','above','below','between',
    'out','off','over','under','again','then','once','here','there','when',
    'where','why','how','all','each','every','more','most','other','some',
    'such','no','nor','only','own','same','than','too','very','just','because',
    's','t','don','won','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ll','m','mightn','mustn','needn','re','shan','shouldn',
    've','wasn','weren','wouldn'
}

# Simple stemmer rules - no download needed
def simple_stem(word):
    suffixes = ['ing', 'tion', 'ness', 'ment', 'able', 'ible', 'er', 'ed', 'ly', 'es', 's']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word

def transform_text(text):
    # Step 1: lowercase
    text = text.lower()
    # Step 2: remove punctuation then split (replaces word_tokenize)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    # Step 3: keep only alphanumeric tokens
    tokens = [t for t in tokens if t.isalnum()]
    # Step 4: remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # Step 5: stem each word
    tokens = [simple_stem(t) for t in tokens]
    return ' '.join(tokens)

data['transformed_text'] = data['Message'].apply(transform_text)
data[['Message', 'transformed_text']].head()

for i in range(5):
    print("Original:", data['Message'][i])
    print("Processed:", data['transformed_text'][i])
    print("-" * 50)

data.head()

data[data['transformed_text'].str.len() == 0]

(data['transformed_text'].str.len() == 0).sum()




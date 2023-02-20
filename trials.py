import csv
import re
import sys
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split


class Song():
    def __init__(self,artist,title,lyrics,link):
        self.artist = artist
        self.title=title
        self.link=link
        self.lyrics=lyrics

class Data:
    bigramCounts={}
    trigramCounts={}
    vocab={}
    model=None
    titles=[]
    wordToIndex=set()
    song_record={}
    probabilities={}
    adj_model={}
    adjective_songs={}

#Stage 1
def corpus_tokenizer(song_file,limit):
    #step 1.2: read the csv to memory
    with open(song_file) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line_count = 0
        titles=[]
        song_record={}
        for row in csv_reader:
            if line_count<limit:
                if line_count == 0:
                    line_count += 1
                    continue
                    #skip header
                else:
                    artist_song= str(row[0].lower().replace(" ","_") + "-" + row[1].lower().replace(" ","_"))

                    # 1.3 tokenize the song titles
                    #Data.titles = Data.titles.append(row[1].lower())
                    tokenize_title = tokenize(row[1])
                    titles.append(tokenize_title)

                    # 1.4 tokenize the song lyrics
                    tokenize_lyrics=tokenize(row[3])
                    tokenize_lyrics.insert(0, "<s>")
                    tokenize_lyrics.append("</s>")
                    # store song record -> Artist, title, lyrics, link
                    new_song=Song(row[0],tokenize_title,tokenize_lyrics,row[2])
                    song_record[artist_song] = new_song
                    line_count += 1
            else:
                break

    Data.titles = list(chain.from_iterable(titles))
    Data.song_record=song_record


#Stage 2
def add_one_trigram_model(limit):
    #step 2.1: Create a vocabulary of words from lyics
    song_record=Data.song_record
    record_limit=limit
    vocab={}
    for song_id in song_record:
        if record_limit<0:
            break
        song =song_record[song_id]
        lyrics=song.lyrics
        for word in lyrics:
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
        record_limit -= 1
    oov_count1=len(vocab)
    vocab = {key:val for key, val in vocab.items() if val > 1}
    vocab["OOV"]=oov_count1-len(vocab)
    Data.vocab = vocab

    # step 2.2 build bigram matrix
    bigramCounts = {}
    record_limit = limit
    for song_id in song_record:
        if record_limit<0:
            break
        song = song_record[song_id]
        lyrics = song.lyrics
        for i in range(1, len(lyrics)):
            if lyrics[i] not in vocab:
                word1="<OOV>"
            else:
                word1=lyrics[i]
            if lyrics[i-1] not in vocab:
                word2 = "<OOV>"
            else:
                word2 = lyrics[i - 1]
            try:
                bigramCounts[word2][word1] += 1
            except KeyError:
                if word2 not in bigramCounts:
                    bigramCounts[word2] = {}
                bigramCounts[word2][word1] = 1

        record_limit -= 1

    Data.bigramCounts = bigramCounts

    # step 2.3 trigram model
    trigramCounts = {}
    # populate trigram
    record_limit = limit

    for song_id in song_record:
        if record_limit<0:
            break
        song = song_record[song_id]
        lyrics = song.lyrics
        for i in range(2, len(lyrics)):
            if lyrics[i] not in vocab:
                word1 = "<OOV>"
            else:
                word1 = lyrics[i]
            if lyrics[i-1] not in vocab:
                word2 = "<OOV>"
            else:
                word2 = lyrics[i - 1]
            if lyrics[i-2] not in vocab:
                word3 = "<OOV>"
            else:
                word3 = lyrics[i - 2]
            try:
                trigramCounts[word3][word2][word1] += 1
            except KeyError:
                if word3 not in trigramCounts:
                    trigramCounts[word3] = {}
                    trigramCounts[word3][word2]={}
                    trigramCounts[word3][word2][word1]=1
                elif word2 not in trigramCounts[word3]:
                    trigramCounts[word3][word2]={}
                    trigramCounts[word3][word2][word1] = 1
                else:
                    trigramCounts[word3][word2][word1] = 1

        record_limit-=1
    Data.trigramCounts = trigramCounts


# Step 2.1, 2.2, and 2.3 modified for language modeling in step 3
def lang_model_maker(lyrics_list):
    # step 2.1: Create a vocabulary of words from lyics
    vocab = {}
    for lyrics in lyrics_list:
        for word in lyrics:
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1

    oov_count1 = len(vocab)
    vocab = {key: val for key, val in vocab.items() if val > 1}
    vocab["OOV"] = oov_count1 - len(vocab)
    vocab = vocab

    # step 2.2 build bigram matrix
    bigramCounts = {}

    for lyrics in lyrics_list:
        for i in range(1, len(lyrics)):
            if lyrics[i] not in vocab:
                word1 = "<OOV>"
            else:
                word1 = lyrics[i]
            if lyrics[i - 1] not in vocab:
                word2 = "<OOV>"
            else:
                word2 = lyrics[i - 1]
            try:
                bigramCounts[word2][word1] += 1
            except KeyError:
                if word2 not in bigramCounts:
                    bigramCounts[word2] = {}
                bigramCounts[word2][word1] = 1

    # step 2.3 trigram model
    trigramCounts = {}
    # populate trigram
    for lyrics in lyrics_list:
        for i in range(2, len(lyrics)):
            if lyrics[i] not in vocab:
                word1 = "<OOV>"
            else:
                word1 = lyrics[i]
            if lyrics[i - 1] not in vocab:
                word2 = "<OOV>"
            else:
                word2 = lyrics[i - 1]
            if lyrics[i - 2] not in vocab:
                word3 = "<OOV>"
            else:
                word3 = lyrics[i - 2]
            try:
                trigramCounts[word3][word2][word1] += 1
            except KeyError:
                if word3 not in trigramCounts:
                    trigramCounts[word3] = {}
                    trigramCounts[word3][word2] = {}
                    trigramCounts[word3][word2][word1] = 1
                elif word2 not in trigramCounts[word3]:
                    trigramCounts[word3][word2] = {}
                    trigramCounts[word3][word2][word1] = 1
                else:
                    trigramCounts[word3][word2][word1] = 1

    return [vocab, bigramCounts, trigramCounts]


#Step 2.4: calculate the probabilities for all potential words
def calculate_all_probability():
    probabilities = {}
    vocab = Data.vocab
    song_record = Data.song_record
    for song_id in song_record:
        song = song_record[song_id]
        lyrics = song.lyrics
        #bigram probability for i => 0,1
        if lyrics[1] in vocab:
            word1=lyrics[1]
            if lyrics[0] not in vocab:
                prob = calculate_probability(word1, [])
                probabilities[word1] = prob
            else:
                word2 = lyrics[0]
                prob = calculate_probability(word1, [word2])
                probabilities[(word2, word1)] = prob
        for i in range(2, len(lyrics)):
            if lyrics[i] not in vocab:
                continue
            else:
                word1 = lyrics[i]
            if lyrics[i-1] not in vocab:
                prob = calculate_probability(word1,[])
                probabilities[word1]=prob
                continue
            else:
                word2 = lyrics[i - 1]
                prob = calculate_probability(word1, [word2])
                probabilities[(word2,word1)] = prob
            if lyrics[i-2] not in vocab:
                continue
            else:
                word3 = lyrics[i - 2]
                prob = calculate_probability(word1, [word3,word2])
                probabilities[(word3, word2, word1)] = prob

    Data.probabilities=probabilities

# Step 2.4 helper method calculate unigram/bigram/trigram probability as required
def calculate_probability(cur_word,previous):
    #previous= [first_prev_word,second_prev_word]

    if len(previous)== 0:
        return Data.bigramCounts[cur_word]["<OOV>"] / len(Data.vocab)

    if len(previous)>1:
        prev_word = previous[1]
    else:
        prev_word = previous[0]

    x=Data.bigramCounts[prev_word][cur_word] + 1
    y=sum(Data.bigramCounts[prev_word].values())
    z=len(Data.vocab.keys())

    bigram_prob = x/(y+z)

    if len(previous) == 1:
        return bigram_prob

    #calculate the trigram probability
    second_prev_word=previous[0]
    if second_prev_word not in Data.vocab:
        second_prev_word = "<OOV>"

    trigram_prob = (Data.trigramCounts[second_prev_word][prev_word][cur_word]+1)/(Data.bigramCounts[second_prev_word][prev_word]+len(Data.vocab.keys()))

    return (bigram_prob + trigram_prob)/ 2


#calculate probability for step 3.4 => modified to
def calculate_all_probability_model(lyrics_list,bigramModel,trigramModel,vocab):
    probabilities = {}
    for lyrics in lyrics_list:
        if lyrics[1] in vocab:
            word1=lyrics[1]
            if lyrics[0] not in vocab:
                prob = calculate_probability_model(word1, [],bigramModel,trigramModel,vocab)
                probabilities[word1] = prob
            else:
                word2 = lyrics[0]
                prob = calculate_probability_model(word1, [word2],bigramModel,trigramModel,vocab)
                probabilities[(word2, word1)] = prob
        for i in range(2, len(lyrics)):
            j=i-2
            if lyrics[i] not in vocab:
                continue
            else:
                word1 = lyrics[i]
            if lyrics[i-1] not in vocab:
                prob = calculate_probability_model(word1,[],bigramModel,trigramModel,vocab)
                probabilities[word1]=prob
                continue
            else:
                word2 = lyrics[i - 1]
                prob = calculate_probability_model(word1, [word2],bigramModel,trigramModel,vocab)
                probabilities[(word2,word1)] = prob
            if lyrics[i-2] not in vocab:
                continue
            else:
                word3 = lyrics[i - 2]
                prob = calculate_probability_model(word1, [word3,word2],bigramModel,trigramModel,vocab)
                probabilities[(word3, word2, word1)] = prob


    return probabilities

#helper function for calculate_all_probability
def calculate_probability_model(cur_word,previous,bigramModel,trigramModel,vocab):

    if len(previous)==0:
        return bigramModel["<OOV>"][cur_word] / len(vocab)

    if len(previous)>1:
        if previous[1] not in vocab:
            prev_word = "<OOV>"
        else:
            prev_word = previous[1]
    else:
        if previous[0] not in vocab:
            prev_word="<OOV>"
        else:
            prev_word = previous[0]

    x=bigramModel[prev_word][cur_word] + 1
    y=sum(bigramModel[prev_word].values())
    z=len(vocab.keys())

    bigram_prob = x/(y+z)

    if len(previous) == 1:
        return bigram_prob

    #calculate the trigram probability
    second_prev_word=previous[0]
    if second_prev_word not in vocab:
        second_prev_word = "<OOV>"

    trigram_prob = (trigramModel[second_prev_word][prev_word][cur_word]+1)/(bigramModel[second_prev_word][prev_word]+len(vocab.keys()))

    return (bigram_prob + trigram_prob)/ 2


# Step 2.4 helper method calculate unigram/bigram/trigram probability as required
def calculate_probability(cur_word,previous):
    #previous= [second_prev_word,first_prev_word]

    if len(previous)==0:
        return Data.bigramCounts[cur_word]["<OOV>"] / len(Data.vocab)

    if len(previous)>1:
        prev_word = previous[1]
        if prev_word not in Data.vocab:
            prev_word="<OOV>"
    else:
        prev_word = previous[0]
        if prev_word not in Data.vocab:
            prev_word = "<OOV>"

    x=Data.bigramCounts[prev_word][cur_word] + 1
    y=sum(Data.bigramCounts[prev_word].values())
    z=len(Data.vocab.keys())

    bigram_prob = x/(y+z)

    if len(previous) == 1:
        return bigram_prob

    #calculate the trigram probability
    second_prev_word=previous[0]
    if second_prev_word not in Data.vocab:
        second_prev_word = "<OOV>"

    trigram_prob = (Data.trigramCounts[second_prev_word][prev_word][cur_word]+1)/(Data.bigramCounts[second_prev_word][prev_word]+len(Data.vocab.keys()))

    return (bigram_prob + trigram_prob)/ 2

#Stage 3
def train_model():
    # step 3.1: raining on 'daily547.conll' and creating a model
    taggedSents = getConllTags('daily547.conll')
    wordToIndex = set()  # maps words to an index
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)  # splits [(w, t), (w, t)] into [w, w], [t, t]
            wordToIndex |= set(words)  # union of the words into the set
    print("  [Read ", len(taggedSents), " Sentences]")
    # turn set into dictionary: word: index
    wordToIndex = {w: i for i, w in enumerate(wordToIndex)}
    wordToIndex["<OOV>"] = len(wordToIndex)
    Data.wordToIndex = wordToIndex;

    # Next, call Feature extraction per sentence
    sentXs = []
    sentYs = []
    print("  [Extracting Features from congll]")
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)
            sentXs.append(getFeaturesForTokens(words, wordToIndex))
            sentYs.append([1 if t == 'A' else 0 for t in tags])

    from sklearn.model_selection import train_test_split
    # flatten by word rather than sent:
    X = [j for i in sentXs for j in i]
    y = [j for i in sentYs for j in i]
    try:
        X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y),
                                                            test_size=0.20,
                                                            random_state=42)
    except ValueError:
        print("[something wong]")
        sys.exit(1)

    # Train the model tagger.
    Data.modelTagger = trainAdjectiveClassifier(X_train, y_train)

    #Step 3.2: Extract features
    titles = Data.titles[0:15000]
    extract_features_from_title = getFeaturesForTokens(titles, wordToIndex)

    #step 3.3
    predicted= Data.modelTagger.predict(np.array(extract_features_from_title))

    seen_adjectives=set()
    adjective_lyrics={}
    adjective_songs={}
    song_record=Data.song_record
    titles=Data.titles

    for i in range(0,len(extract_features_from_title)):
        if predicted[i] == 1:
            adj = titles[i]
            if adj not in seen_adjectives:
                seen_adjectives.add(adj)
                all_lyrics=[]
                all_songs=[]
                for song_id in song_record:
                    song = song_record[song_id]
                    title = song.title
                    if adj in title:
                        all_songs.append(song_id)
                        all_lyrics.append(song.lyrics)
                adjective_lyrics[adj] = all_lyrics
                adjective_songs[adj]=all_songs

    Data.adjective_songs=adjective_songs
    #adjective_lyrics = {key:val for key, val in adjective_lyrics.items() if len(val) > 9}

    adj_model={}
    for adj in adjective_lyrics:
        new_lang_model = lang_model_maker(adjective_lyrics[adj]) #vocab,bigram,trigram
        probabilities= calculate_all_probability_model(adjective_lyrics[adj],new_lang_model[1],new_lang_model[2],new_lang_model[0])
        adj_model[adj] = [new_lang_model, probabilities]

    Data.adj_model=adj_model

#Step 3.5
def generate_lyics(adj):
    model=Data.adj_model[adj][0]
    bigram = model[1]
    probabilities= Data.adj_model[adj][1]
    generated_lyrics=[]

    # choose words to generate
    start = True
    length = 0
    word = "foo"

    while word != "</s>" and length < 32:
        if start:
            start=False
            length += 1
            word = "<s>"
            generated_lyrics.append(word)
            next_words = bigram[word].keys()

        prob_list = []

        for next_word in next_words:
            try:
               prob = probabilities[(word,next_word)]
            except KeyError:
               try:
                   prob=probabilities[(word,"<OOV>")]
               except KeyError:
                   prob=0
            prob_list.append(prob)

        #renormalize probabilities
        tot_prob=sum(prob_list)
        if tot_prob==0:
            continue
        new_prob=[]
        for prob in prob_list:
            new_prob.append(prob/tot_prob)

        rand_words = list(np.random.choice(list(next_words), 1, p=new_prob))
        word=rand_words[0]
        while word == "<OOV>":
            word = np.random.choice(next_words, 1, p=prob_list)
        generated_lyrics.append(word)
        length += 1
        if word != "</s>":
            next_words = bigram[word].keys()
        else:
            generated_lyrics.append("</s>")

    return generated_lyrics

def getConllTags(filename):
    # input: filename for a conll style parts of speech tagged file
    # output: a list of list of tuples
    #        representing [[[word1, tag1], [word2, tag2]]]
    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f:
            wordtag = wordtag.strip()
            if wordtag:  # still reading current sentence
                (word, tag) = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word, tag))
            else:  # new sentence
                wordTagsPerSent.append([])
                sentNum += 1
    return wordTagsPerSent


def getFeaturesForTokens(tokens, wordToIndex):
    # input: tokens: a list of tokens,
    # wordToIndex: dict mapping 'word' to an index in the feature list.
    # output: list of lists (or np.array) of k feature values for the given target

    num_words = len(tokens)
    featuresPerTarget = list()  # holds arrays of feature per word
    word_size = len(wordToIndex)

    for targetI in range(num_words):
        # <FILL IN>
        one_hot = ([0] * 3 * word_size)
        try:
            one_hot[wordToIndex[tokens[targetI]]] = 1
        except KeyError:
            one_hot[wordToIndex["<OOV>"]] = 1
        vowels, consonants = 0, 0

        if targetI > 0:
            if tokens[targetI - 1] not in wordToIndex:
                one_hot[wordToIndex["<OOV>"] + word_size] = 1
            else:
                one_hot[wordToIndex[tokens[targetI - 1]] + word_size] = 1

        if targetI < num_words - 1:
            if tokens[targetI + 1] not in wordToIndex:
                one_hot[wordToIndex["<OOV>"] + 2 * word_size] = 1
            else:
                one_hot[wordToIndex[tokens[targetI + 1]] + 2 * word_size] = 1

        word = tokens[targetI]
        for char in word:
            if char in "aeiouy":
                vowels += 1
            else:
                consonants += 1

        one_hot.append(vowels)
        one_hot.append(consonants)
        featuresPerTarget.append(one_hot)

    return featuresPerTarget  # a (num_words x k) matrix


from sklearn.linear_model import LogisticRegression

def trainAdjectiveClassifier(features, adjs):
    # inputs: features: feature vectors (i.e. X)
    #        adjs: whether adjective or not: [0, 1] (i.e. y)
    # output: model -- a trained sklearn.linear_model.LogisticRegression object
    # <FILL IN>
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        adjs,
                                                        test_size=0.10,
                                                        random_state=42)

    X_trainsub, X_dev, y_trainsub, y_dev = train_test_split(X_train, y_train,
                                                            test_size=0.20, random_state=22)
    # train:
    # check the best penalty
    C = [10, 100, 1000, 10000]
    max_acc = 0
    best_c = 0

    for c in C:
        cur_model = LogisticRegression(C=1, penalty="l1", solver='liblinear')
        cur_model.fit(X_trainsub, y_trainsub)
        y_pred = cur_model.predict(X_dev)
        acc = (1 - np.sum(np.abs(y_pred - y_dev)) / len(y_dev))
        if acc > max_acc:
            best_c = c

    # y_pred = classifier.predict(X_test)
    # y_prob = classifier.predict_proba(X_test)
    model = LogisticRegression(C=best_c, penalty="l1", solver="liblinear")
    model.fit(X_train, y_train)
    return model

def tokenize(sent):
    # input: a single sentence as a string.
    # output: a list of each "word" in the text
    # must use regular expressions
    # <FILL IN>
    words = re.split(' ', sent)
    tokens = []
    for word in words:
        word=word.lower()
        #^([A-Z].)+$
        if re.match(r'\n[A-Za-z]*',word):
            if len(word) > 1:
                tokens.append("<newline>")
                if tokenize(word[1:]) != []:
                    tokens.append(tokenize(word[1:])[0])
            else:
                tokens.append("<newline>")
        elif re.match(r'[A-Za-z]*\n',word):
            if len(word) > 1:
                tokens.append(tokenize(word[0:len(word)-1])[0])
                tokens.append("<newline>")
            else:
                tokens.append("<newline>")
        elif re.match(r'\([A-Za-z]*', word):
            if len(word) > 1:
                if word[len(word)-1] == ')':
                    tokens.append(word[1:len(word)-1])
                else:
                    tokens.append(word[1:len(word)])
        elif re.match(r'[A-Za-z]*\)', word):
            if len(word)>1:
                tokens.append(word[0:len(word)-1])
        elif word.isalpha():
            tokens.append(word)
        elif re.match(r'[A-Za-z]*[?.!]$', word):
            tokens.append(word[0:len(word)-1])
            tokens.append(word[len(word)-1])
        elif re.match(r'^[#,@]',word):
            tokens.append(word)
        elif re.match(r'(?:(?<=\.|\s)[A-Z]\.)+',word):
            tokens.append(word)
        elif re.match(r"(?!='.*')\b[\w']+\b",word):
            tokens.append(word)
        else:
            tokens.append(word)

    tokens = [val for val in tokens if len(val) >= 1]
    return tokens

def main():
    print(" 'system start")

    #initialize the song records and tokenize title and lyrics for all songs
    #57651
    corpus_tokenizer('songdata.csv',57651)

    #Checkpoint 1
    print("Checkpoint 1: printing tokenized song titles and lyrics")

    print("Title:")
    print(Data.song_record["abba-burning_my_bridges"].title)
    print("Lyrics:")
    print(Data.song_record["abba-burning_my_bridges"].lyrics)

    print("Title:")
    try:
        title=Data.song_record["beach_boys-do_you_remember?"].title
    except KeyError:
        title="song not found"
    print(title)
    print("Lyrics:")
    #print(Data.song_record["beach_boys-do_you_remember?"].lyrics)
    try:
        title=Data.song_record["beach_boys-do_you_remember?"].lyrics
    except KeyError:
        title="song not found"

    print("Title:")
    try:
        print(Data.song_record["avril_lavigne-5,_4,_3,_2,_1_(countdown)"].title)
    except KeyError:
        print("song not found")
    print("Lyrics:")
    try:
        print(Data.song_record["avril_lavigne-5,_4,_3,_2,_1_(countdown)"].lyrics)
    except:
        print("song not found")

    print("Title:")
    try:
        print(Data.song_record["michael_buble-l-o-v-e"].title)
    except:
        print("song not found")
    print("Lyrics:")
    try:
        print(Data.song_record["michael_buble-l-o-v-e"].lyrics)
    except:
        print("song not found")
    
    #Checkpoint 2
    add_one_trigram_model(5000)
    print("p(wi= “you” | wi - 2 = “I”, wi - 1 = “love”.) = " + str(calculate_probability("you",["i","love"])))
    print("p(wi=”special” | wi - 1 =”midnight”) = " + str(calculate_probability("special",["midnight"])))
    print("p(wi=”special” | wi - 1 =”very”) = " + str(calculate_probability("special",["very"])))
   # print("p(wi=”special” | wi - 2 =”something ”, wi - 1 =”very”)) = " + str(calculate_probability("special",["something","very"])))
    try:
        print("p(wi=”funny” | wi - 2 =”something ”, wi - 1 =”very”) = " + str(calculate_probability("funny",["something", "very"])))
    except KeyError:
        print("p(wi=”funny” | wi - 2 =”something ”, wi - 1 =”very” ) couldn't be found")

    #Checkpoint 3
    train_model()
    print("Adjective associated with artist-song")
    print("good: ")
    print(Data.adjective_songs["good"])
    print("happy: ")
    print(Data.adjective_songs["happy"])
    print("afraid: ")
#    print(Data.adjective_songs["afraid"])
    print("red: ")
    try:
        red=Data.adjective_songs["red"]
    except KeyError:
        red= "Red not tagged as an adjective"
    print(red)
    print("blue: ")
    try:
        blue=Data.adjective_songs["blue"]
    except KeyError:
        blue= "blue not tagged as an adjective"
    print(blue)

    print("Generate 3 lyrics for each adjective")

    print("Lyrics for good: ")
    i=0
    while i<3:
        print(generate_lyics("good"))
        i+=1

    print("Lyrics for happy: ")
    i = 0
    while i < 3:
        print(generate_lyics("happy"))
        i += 1

    print("Lyrics for afraid: ")
    if "afraid" in Data.adj_model:
        i = 0
        while i < 3:
            print(generate_lyics("afraid"))
            i += 1
    else:
        print("afraid not tagged as an adjective")

    print("Lyrics for red: ")
    if "red" in Data.adj_model:
        i = 0
        while i < 3:
            print(generate_lyics("red"))
            i += 1
    else:
        print("red not tagged as an adjective")

    print("Lyrics for blue: ")
    if "blue" in Data.adj_model:
        i = 0
        while i < 3:
            print(generate_lyics("blue"))
            i += 1
    else:
        print("blue not tagged as an adjective")

    print("'system end")


main()
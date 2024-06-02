#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#Project: Topic Modeling on Candidate Debate of USA 2016 Election
#Author : Dewan F. Wahid
#Supervisor's: Dr. Paramjit Gill,Associate Professor of Statistics, UBC Okanagan
#Date: Oct 05, 2015
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# Packages ..............................................................................
install.packages("lda")
install.packages("tm")
install.packages("SnowballC")
install.packages("wordcloud")

# USING LIBRARIES #
library(lda)
library(tm)
library(wordcloud)
#.......................................................................................


# DATA READING FROM LOCAL TEXT  #
TEXTFILE = "RepCandidateDebate_Sep16_2015.txt"
textDocs <- readLines(TEXTFILE)
length(textDocs)

# CLEAN THE INPUT TEXT #

# convert text to corpus
corpus <- Corpus(VectorSource(textDocs))

# standardize case
corpus <- tm_map(corpus, tolower)

# remove stopwords / numbers / punctuation / whitespace
stop_words <- c(stopwords('english'), "wisc")
corpus <- tm_map(corpus, removeWords, stop_words)

# remove numbers / punctuation / strip whitespace
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)

# strip white spaces
corpus <- tm_map(corpus, stripWhitespace)



# steaming corpus
corpus <- tm_map(corpus, stemDocument)

lapply(corpus[1:2], as.character)


# convert corpus back to character vector for lexicalizing
text <- as.character(corpus)

# CREATE / FILTER LEXICON #

# lexicalize text
corpus <- lexicalize(text, lower=TRUE)

# deleted particular words from corpus$vocab
corpus$vocab[3493] = "a"
corpus$vocab[3494] = "b"
corpus$vocab[3447] = "c"
corpus$vocab[2955] = "d"
corpus$vocab[2747] = "e"
corpus$vocab[2748] = "f"
corpus$vocab[2523] = "g"
corpus$vocab[2383] = "h"
corpus$vocab[2384] = "i"
corpus$vocab[2139] = "j"
corpus$vocab[1943] = "k"
corpus$vocab[1657] = "l"
corpus$vocab[1401] = "m"
corpus$vocab[1402] = "n"
corpus$vocab[549] = "o"
corpus$vocab[550] = "p"
corpus$vocab[551] = "q"
corpus$vocab[552] = "r"
corpus$vocab[1] = "s"
corpus$vocab[3361] = "t"
corpus$vocab[3047] = "u" 


# only keep words that appear at least twice.
to.keep <- corpus$vocab[word.counts(corpus$documents, corpus$vocab) >= 2]

# re-lexicalize, using this subsetted vocabulary
finalCorpus <- lexicalize(corpus, lower=TRUE, vocab=to.keep)

# FIT TOPICS #

##This sets the random seed
set.seed(12345) 

# gibbs sampling
# K is the number of topics
K <- 5
result <- lda.collapsed.gibbs.sampler(
              finalCorpus$documents, #This is the set of documents
              K,   #This is the number of clusters
              finalCorpus$vocab, #This is the vocab set
              25,  #These are additional model parameters
              0.1, #Alpha
              0.1 ) #Eta 

# # PREPARE OUTPUT #

# top words by document
predictions <- t(predictive.distribution(result$document_sums, result$topics, 0.1, 0.1))
document_words <- data.frame(top.topic.words(predictions, n_topic_words, by.score = TRUE))

# top words by topic
topic_words <- data.frame(top.topic.words(result$topics, num.words = 5, by.score = TRUE))
names(topic_words) <- paste0("topic_", 1:K)

# topics by documents stats
raw <- as.data.frame(t(result$document_sums))
names(raw) <- 1:K
n_docs  <- nrow(raw)
topics <- data.frame(id = ids, matrix(0, nrow = n_docs, ncol=2*K))
names(topics) <- c("id", paste0("n_topic_", 1:K), paste0("p_topic_", 1:K))



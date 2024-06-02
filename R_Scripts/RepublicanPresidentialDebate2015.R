#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#Project: Topic Modeling on Candidate Debate of USA 2016 Election
#Author : Dewan F. Wahid
#Supervisor's: Dr. Paramjit Gill,Associate Professor of Statistics, UBC Okanagan
#Date: Oct 05, 2015
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#Packages
install.packages("lda")
install.packages("topicmodels")
install.packages("tm")
install.packages("SnowballC")
install.packages("wordcloud")
install.packages("topicmodels")
install.packages("qdapRegex")
install.packages("LDAvis")
install.packages("RWeka")

#Using Library
library(lda)
library(tm)
library(NLP)
library(RColorBrewer)
library(topicmodels)
library(ggplot2) 
library(RWeka)
library(LDAvis)


## READING RAW DATA FROM A SINGLE TXT FILE#..............................................
# Reading raw data from local text file 
#TEXTFILE = "RepCandidateDebate_Sep16_2015.txt";
#doc.text <- readLines(TEXTFILE);
#length(doc.text);




## READING RAW DTATA FROM A FOLDER: ...............................................................
folder1 <- file.path("StateOfUnionAddress", "Bush_Junior")  
doc.corpus <- Corpus(DirSource(folder1))
summary(doc.corpus)



## CLEANING CORPUS ##..............................................................................
#Remove common english stop words, number and punctuation
doc.corpus <- tm_map(doc.corpus, content_transformer(tolower))        # convert upper case to lower case 
doc.corpus <- tm_map(doc.corpus, removePunctuation);                  # remove punctuations
doc.corpus <- tm_map(doc.corpus, removeNumbers);                      # remove numbers
myStopWords <- c(stopwords("english"))
myStopWords <- c("and", "are", "for", "have", "not", "that", "this", "what", 
                 "with", "you", "going", "now", "name" , "and", "think",
                 "dont", "get", "know", "need", "one", "say", "year",  
                 "want", "well", "will", "right", "said", "can", "just", 
                 "let", "make", "thing", "time", "that", "year", "talk",
                 "come", "day", "didnt", "first", "give", "got", "everi",
                 "issu", "ive", "lot", "like", "look", "put", "two", "tell",
                 "use", "your", "abl", "ago", "agr", "allow", "also", "anybodi","around",
                 "ask", "cant", "came", "rai", "tri", "weve", "youv" , "send", "solv",
                 "somebodi", "someon", "theyr", "yes", "six", "sinc", "three", "wouldnt" ,
                 "wont", "went", "show", "youll","within", "whos", "thei" ,"thei","theyll",
                 "theyv", "theth", "thethat","thejak", "thethank", "sorri", "simpl", "simpli",
                 "see", "seem", "seen", "sen", "second", "where", "whether", "whi", "which",
                 "while", "yeah", "your", "youv", "yet", "will", "whi", "who", "whole", "was", 
                 "wasnt", "what", "when", "were", "welcom", "veri" , "theyr", "theyv", "thank",
                 "that","the", "than", "thank", "those", "their", "though", "may", "mayb", "might",
                 "your","youv", "but","all", "they" ,"our", "there", "would", "too", "thewel" , "these",
                 "then", "these", "them", "seven", "shes","she", "should", "shouldnt","sir", "his", 
                 "him", "tapper", "has", "done", "that", "thing", "here", "been", "did", "becaus",
                 "from" , "about", "becaus", "jindal", "her", "hewitt", "some", "reagan",
                 "out", "back", "way", "unit")

doc.corpus <- tm_map(doc.corpus, removeWords, myStopWords); 
doc.corpus <- tm_map(doc.corpus, PlainTextDocument);
doc.corpus <- tm_map(doc.corpus, stripWhitespace);


# CREATE DOCUMENT TERMS MATRIX (DTM)and TERMS DOCUMENT MATRIX (TDM)
# 'control = list(minWordLength = 4)' only allows words with >= lenght 4 
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2))



#doc.DTM <- DocumentTermMatrix(doc.corpus, control = list(tokenize = BigramTokenizer, minWordLength = 3));

doc.DTM <- DocumentTermMatrix(doc.corpus, control = list(minWordLength = 3));



doc.TDM <- TermDocumentMatrix(doc.corpus, control = list(minWordLength = 3));

# Removing the rows without any entries from 'doc.DTM'
rowTotals <- apply(doc.DTM , 1, sum) #Find the sum of words in each Document
doc.DTM.new   <- doc.DTM[rowTotals> 0, ] 


# Check words with frequency 
#findFreqTerms(doc.DTM.new, lowfreq=5);

# find the probability a word is associated
#findAssocs(doc.DTM.new, 'obamacar', 0.2);

# Check individual documents
#writeLines(as.character(doc.corpus[2]))

# Plot words frequency
freq <- sort(colSums(as.matrix(doc.DTM.new)), decreasing=TRUE)  
wf <- data.frame(word=names(freq), freq=freq)   
p <- ggplot(subset(wf, freq>50), aes(word, freq))    
p <- p + geom_bar(stat="identity")   
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1)) 
p

# Words Clouds 
set.seed(142)   
wordcloud(names(freq), freq, colors=c(3,4), random.color=TRUE, min.freq=20)   


#................................................................................................



#>>>>>>>>>>>>>>>>> LATENT DIRICHLET ALLOCATION (LDA) -using 'topicmodels' package <<<<<<<<<<<<<<<<<<<<<<

# Number of topics 
k <- 3
SEED <- 1234;

# Topic Modeling
doc.TM <- list(VEM = LDA(doc.DTM.new, k = k, control = list(seed = SEED)),
               VEM_fixed = LDA(doc.DTM.new, k = k,
                                 control = list(estimate.alpha = FALSE, seed = SEED)),
               Gibbs = LDA(doc.DTM.new, k = k, method = "Gibbs",
                             control = list(seed = SEED, burnin = 1000,
                                               thin = 100, iter = 1000)),
               CTM = CTM(doc.DTM.new, k = k,
                           control = list(seed = SEED,
                                           var = list(tol = 10^-4), em = list(tol = 10^-3))));


# Top 10 terms for each topic in LDA
Terms = terms(doc.TM[["VEM"]], 15);
Terms

# Finding most frequent words
my_topics <- topics(doc.TM[["VEM"]]);
most_frequent = which.max(tabulate(my_topics));
terms(doc.TM[["VEM"]], 10)[, most_frequent]








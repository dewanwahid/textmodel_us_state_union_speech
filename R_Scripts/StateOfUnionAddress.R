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
#install.packages("LDAvis")
install.packages("RWeka")
install.packages("rvest")


#http://francojc.githum.io/we-scraping-with-rvest


#Using Library
library(rvest)
library(lda)
library(tm)
library(NLP)
library(RColorBrewer)
library(topicmodels)
library(ggplot2) 
library(RWeka)
library(LDAvis)



## READING RAW DTATA FROM A FOLDER: ...............................................................
folder1 <- file.path("StateOfUnionAddress", "Reagan-Obama") 

## CREATING CORPUS##.....................................................................
doc.corpus <- Corpus(DirSource(folder1))
summary(doc.corpus)



## CLEANING CORPUS ##..............................................................................
#Remove common english stop words, number and punctuation
doc.corpus <- tm_map(doc.corpus, content_transformer(tolower))        # convert upper case to lower case 
doc.corpus <- tm_map(doc.corpus, removePunctuation);                  # remove punctuations
doc.corpus <- tm_map(doc.corpus, removeNumbers);                      # remove numbers
#doc.corpus <- tm_map(doc.corpus, stemDocument)
myStopWords <- c(stopwords("english"))
myStopWords <- c("and", "are", "for", "have", "not", "that", "this", "what", 
                 "with", "you", "going", "now", "name" , "and", "think",
                 "dont", "get", "know", "need", "one", "say", "year",  
                 "want", "well", "will", "right", "said", "can", "just", 
                 "let", "america", "american", "americans", "must", "more", 
                 "many", "congress", "people", "years", "world", "all", "also", "ask", "because", 
                 "but", "care", "country", "every", "from", "good", "great", "has", "help",
                 "its", "make", "nation", "nations", "new", "other", "our", "over", "own", "out",
                 "then", "than", "the", "them", "there", "they", "those", "these", "was", "were",
                 "would", "could", "when", "who", "work", "yet", "your", "there", "should", "time", 
                 "tonight", "their", "act", "govenment", "united", 
                 "had", "weve", "strog", "way", "into", "show", "honor","meet", "life", "back",
                 "pass", "while", "cause", "clear", "join", "too", "high", "here", "lives", 
                 "responsibility", "still", "end", "day", "see", "without", "come", "hope", "take",
                 "sure", "first", "down", "give", "serve", "plan", "live", "thank", "govemment", 
                 "small", "set", 'september', "given", "human", "any", "much", "friends","use",
                 "needs", "history", "relief", "leasers", "leaders", "through", "free", "about", 
                 "keep", "where", "best", "last", "two", "fellow", "build", "lead", "east",
                 "men", "made", "working", "better", "goal", "across", "against", "some", "funding", 
                 "members", "middle", "billion","even", "propose", "continue", "growing", "refrom",
                 "future", "his", "states", "only", "some", "rercent", "government", "support","been", 
                 "never", "next","together", "system","support", "women", "iraqi", "past",
                 "empower", "hopeful", "like","both", "keeping", "reform", "hussein", "agreement","most" ,
                 "schools", "security", "against", "some", "funding", "members", "middle", "billion",
                 "even", "propose", "continue", "growing", "reform", "future", "his", "states", "only",
                 "percent", "insurance",  "iraq", "ive", "which", "believe",  "before", "left","increase",
                 "families", "full", "lets", "ago", "very" , "used","home" , "today", "tell" , "things", 
                 "federal",  "put", "thing","after", "big", "change", "kids", "theres", "struggle", 
                 "cannot", "clean", "americas", "union", "change", "kids", "theyre", "thats", "how", 
                 "why", "she", "how", "job", "done", "ever", "her", "create", "million", "cant", "deficit",
                 "bill", "ourselves", "spending", "higher", "making","deserve", "already", "recovery",
                 "cuts", "finally", "pretect", "again","issue","another", "since", "doesnt", "same", 
                 "community", "did", "yong", "may", "paid", "instead", "class", "away", "each", "youre", 
                 "nearly", "office", "again","values", "took", "chance", "single", "afford", "stronger", 
                 "share", "times", "around", "young", "helping", "fact", "bring", "fair", "loughter",
                  "long","send","rules", "democrats", "republican","agree", "millions","face","behind",
                 "win","willing", "workers","hard","jobs", "republicans", "economy", "college", "energy",
                 "education", "tax", "program", "cut", "something", "ought", "challenge", "president", 
                 "parent", "laughter", "businesses",  "century", "companies","success", "always", "off",
                 "built", "opportunity", "companies")

doc.corpus <- tm_map(doc.corpus, removeWords, myStopWords); 
doc.corpus <- tm_map(doc.corpus, PlainTextDocument);
doc.corpus <- tm_map(doc.corpus, stripWhitespace);


# CREATE DOCUMENT TERMS MATRIX (DTM)and TERMS DOCUMENT MATRIX (TDM)
# 'control = list(minWordLength = 4)' only allows words with >= lenght 4 
#BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2))



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
p <- ggplot(subset(wf, freq>70), aes(word, freq))    
p <- p + geom_bar(stat="identity")   
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1)) 
p

# Words Clouds 
set.seed(142)   
wordcloud(names(freq), freq, colors=c(3,4), random.color=TRUE, min.freq=20)   


#................................................................................................
par(mfrow = c(1,2))


#>>>>>>>>>>>>>>>>> LATENT DIRICHLET ALLOCATION (LDA) -using 'topicmodels' package <<<<<<<<<<<<<<<<<<<<<<

# Number of topics 
k <- 5
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


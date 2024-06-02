

#>>>>>>>>>>>>>>>>>>>>>>>>>LDA (using lda package)<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# convert corpus back to character vector for lexicalizing
doc.text.clean <- as.character(doc.corpus)

##Latent Dirichlet Allocation
doc.corpus.clean <- lexicalize(doc.corpus, lower=TRUE)

##Check the vocabulary set
doc.corpus.clean $vocab

## Get the word counts.
wc <- word.counts(doc.corpus.clean, vocab = NULL)

filtered <- filter.words(doc.corpus.clean, as.numeric(names(wc)[wc <= 4]))
## Only keep words that appear at least twice:
to.keep <- doc.corpus.clean $vocab[word.counts(doc.corpus.clean$documents, doc.corpus.clean$vocab) >= 15]

## Re-lexicalize, using this subsetted vocabulary
doc.corpus.clean.final <- lexicalize(doc.corpus.clean, lower=TRUE, vocab=to.keep)



##This sets the random seed
set.seed(12345) 

#Now we run the Latent Dirichlet Allocation Model
K <- 3
result <- lda.collapsed.gibbs.sampler(
  doc.corpus.clean.final$documents, #This is the set of documents
  K, #This is the number of clusters
  doc.corpus.clean.final$vocab, #This is the vocab set
  25, #These are additional model parameters
  0.1,
  0.1)


top.words <- top.topic.words(result$topics, 2, by.score=TRUE)
#The "5" is the number of labels
top.words
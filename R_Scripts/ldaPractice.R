#...................................................
#Name:Latent Dirichlet Allocation
#Author: Dewan Ferdous Wahid
#Date: 21-09-2015
#...................................................


##Installing packages 
install.packages("lda")

##Improting Data for library
library(lda)
data("cora.documents")

##Just use a small subset for the example.
corpus <- cora.documents[1:6]

## Get the word counts.
wc <- word.counts(corpus)

## Only keep the words which occur more than 4 times.
#filtered <- filter.words(corpus, as.numeric(names(wc)[wc <= 4]))

## Shift the second half of the corpus.
#shifted <- shift.word.indices(filtered[4:6], 100)


## Combine the unshifted documents and the shifted documents.
#concatenate.documents(filtered[1:3], shifted)

#lda.collapsed.gibbs.sampler(documents, K, vocab, num.iterations, alpha,
                            #eta, initial = NULL, burnin = NULL, compute.log.likelihood = FALSE,
                            #trace = 0L, freeze.topics = FALSE)

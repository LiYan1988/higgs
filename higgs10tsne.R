library(Rtsne)
setwd("C:\\Users\\lyaa\\Documents\\higgs_all\\higgs")
train = read.csv("..\\train.csv", stringsAsFactors=FALSE)[, -1]
test = read.csv("..\\test.csv", stringsAsFactors = F)[,-1]
x = rbind(train[,1:30], test)

set.seed(1234)
tsne_out <- Rtsne(as.matrix(x), check_duplicates = FALSE, pca = TRUE, 
                        max_iter = 1000, perplexity=30, theta=0.5, dims=2, verbose=TRUE)

write.csv(x = tsne_out$Y, file = 'tsne2all.csv', quote = FALSE, row.names = FALSE)

#! /usr/bin/env Rscript

x <- read.csv('neighborsSqDist.csv', row.names='Particles')
pdf('neighborsSqDist.pdf')
barplot(t(as.matrix(x)), beside=T, col=rainbow(3),
        main='Squared Distance to Neighbors',
        xlab='Number of Particles',
        ylab='Time per Neighbor (ns)')
legend(1, 32, names(x), fill=rainbow(3))
dev.off()

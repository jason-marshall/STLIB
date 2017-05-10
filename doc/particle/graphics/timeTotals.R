#! /usr/bin/env Rscript

x <- read.csv('timeTotals.csv', row.names='Cores')
# Ignore the costs to partition and distribute.
x['Partition'] <- NULL
x['Distribute'] <- NULL
pdf('timeTotals.pdf')
barplot(t(as.matrix(x)), beside=T, col=rainbow(ncol(x)),
        main='Total Time',
        xlab='Number of Cores',
        ylab='Time (s)')
legend(1, 20, names(x), fill=rainbow(ncol(x)))
dev.off()

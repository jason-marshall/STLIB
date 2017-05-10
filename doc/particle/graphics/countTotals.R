#! /usr/bin/env Rscript

x <- read.csv('countTotals.csv', row.names='Cores')

pdf('occupancy.pdf')
barplot(t(x[3]),
        main='Cell Occupancy',
        xlab='Number of Cores',
        ylab='Number of Particles')
dev.off()

pdf('exchangeCount.pdf')
barplot(t(x[4]),
        main='Exchange Count',
        xlab='Number of Cores',
        ylab='Number of Particles')
dev.off()

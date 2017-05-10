#! /usr/bin/env Rscript

data <- read.csv('randomMpi4Time.csv', row.names='Cores')
x <- data
png('randomMpi4TimeAll.png')
barplot(t(as.matrix(x)), beside=T, col=rainbow(ncol(x)),
        main='Total Time',
        xlab='Number of Cores',
        ylab='Time (s)')
legend(1, 40, names(x), fill=rainbow(ncol(x)))
dev.off()

## Keep only repairing costs.
x <- data
x['MoveParticles'] <- NULL
x['CountNeighbors'] <- NULL

png('randomMpi4TimeRepair.png')
barplot(t(as.matrix(x)), beside=T, col=rainbow(ncol(x)),
        main='Total Time',
        xlab='Number of Cores',
        ylab='Time (s)')
legend(1, 15, names(x), fill=rainbow(ncol(x)))
dev.off()


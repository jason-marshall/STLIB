#! /usr/bin/env Rscript

data <- read.csv('randomSharedTime.csv', row.names='Cores')
x <- data
png('randomSharedTimeAll.png')
barplot(t(as.matrix(x)), beside=T, col=rainbow(ncol(x)),
        main='Total Time',
        xlab='Number of Cores',
        ylab='Time (s)')
legend(20, 1200, names(x), fill=rainbow(ncol(x)))
dev.off()

for (i in rownames(x)) {
  x[i,] <- x[i,] * as.numeric(i)
}
png('randomSharedScaledAll.png')
barplot(t(as.matrix(x)), beside=T, col=rainbow(ncol(x)),
        main='Scaled Time',
        xlab='Number of Cores',
        ylab='Time * Cores (s)')
legend(1, 1000, names(x), fill=rainbow(ncol(x)))
dev.off()

## Keep only repairing costs.
x <- data
x['MoveParticles'] <- NULL
x['CountNeighbors'] <- NULL

png('randomSharedTimeRepair.png')
barplot(t(as.matrix(x)), beside=T, col=rainbow(ncol(x)),
        main='Total Time',
        xlab='Number of Cores',
        ylab='Time (s)')
legend(16, 100, names(x), fill=rainbow(ncol(x)))
dev.off()

for (i in rownames(x)) {
  x[i,] <- x[i,] * as.numeric(i)
}
png('randomSharedScaledRepair.png')
barplot(t(as.matrix(x)), beside=T, col=rainbow(ncol(x)),
        main='Scaled Time',
        xlab='Number of Cores',
        ylab='Time * Cores (s)')
legend(1, 200, names(x), fill=rainbow(ncol(x)))
dev.off()

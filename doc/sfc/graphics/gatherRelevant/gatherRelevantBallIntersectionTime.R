#! /usr/bin/env Rscript

x <- read.csv('gatherRelevantBallIntersectionTime.csv', row.names='Cores')
png('GatherRelevantBallIntersectionTime.png')
barplot(t(as.matrix(x)), beside=T, col=rainbow(ncol(x)),
        main='Gather Relevant for Ball Intersection',
        xlab='Number of Cores',
        ylab='Time (s)')
#        log='y')
legend(1, 3, names(x), fill=rainbow(ncol(x)))
dev.off()

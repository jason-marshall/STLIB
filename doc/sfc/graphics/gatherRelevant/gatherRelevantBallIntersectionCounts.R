#! /usr/bin/env Rscript

x <- read.csv('gatherRelevantBallIntersectionCounts.csv', row.names='Cores')
x['P2P_Comm'] = x['P2P_Send'] + x['P2P_Recv']
x['P2P_Send'] <- NULL
x['P2P_Recv'] <- NULL
x['Ring_Comm'] = x['Ring_Send'] + x['Ring_Recv']
x['Ring_Send'] <- NULL
x['Ring_Recv'] <- NULL
png('GatherRelevantBallIntersectionCounts.png')
barplot(t(as.matrix(x)), beside=T, col=rainbow(ncol(x)),
        main='Gather Relevant for Ball Intersection',
        xlab='Number of Cores',
        ylab='Storage (bytes)')
legend(1, 3e7, names(x), fill=rainbow(ncol(x)))
dev.off()

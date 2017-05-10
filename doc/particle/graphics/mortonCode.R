#! /usr/bin/env Rscript

x <- read.csv('mortonCode.csv', header=TRUE, row.names=1)
x <- t(x)
rownames(x) <- 0:(nrow(x)-1)

pdf('coordinatesToCode.pdf')
barplot(x[,'CoordinatesToCode'],
        main='Discrete Coordinates to Morton Code',
        xlab='Number of Levels',
        ylab='Time (ns)')
dev.off()

pdf('codeToCoordinates.pdf')
barplot(x[,'CodeToCoordinates'],
        main='Morton Code to Discrete Coordinates',
        xlab='Number of Levels',
        ylab='Time (ns)')
dev.off()

pdf('cartesianToCode.pdf')
barplot(x[,'CartesianToCode'],
        main='Cartesian Point to Morton Code',
        xlab='Number of Levels',
        ylab='Time (ns)')
dev.off()

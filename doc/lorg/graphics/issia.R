#! /usr/bin/env Rscript

x <- read.csv('issia.csv', row.names='Tets')

# Ordering.
y = x[c('Axis', 'Mort8', 'Mort16', 'Mort32', 'Mort64')]
pdf('order.pdf')
barplot(t(as.matrix(y)), beside=T, col=rainbow(ncol(y)),
        main='Ordering',
        xlab='Number of Tets',
        ylab='Time per Tet (ns)')
legend(1, 800, names(y), fill=rainbow(ncol(y)))
dev.off()

# Content.
y = x[c('ContRand', 'ContAxis', 'ContMort8', 'ContMort16', 'ContMort32',
  'ContMort64')]
pdf('content.pdf')
barplot(t(as.matrix(y)), beside=T, col=rainbow(ncol(y)),
        main='Calculate Volume',
        xlab='Number of Tets',
        ylab='Time per Tet (ns)')
legend(1, 70, names(y), fill=rainbow(ncol(y)))
dev.off()

# Laplacian.
y = x[c('LapRand', 'LapAxis', 'LapMort8', 'LapMort16', 'LapMort32',
  'LapMort64')]
pdf('laplacian.pdf')
barplot(t(as.matrix(y)), beside=T, col=rainbow(ncol(y)),
        main='Laplacian Smoothing',
        xlab='Number of Tets',
        ylab='Time per Tet (ns)')
legend(1, 150, names(y), fill=rainbow(ncol(y)))
dev.off()

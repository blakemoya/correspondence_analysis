library(ca)

cont <- read.csv('src.csv', row.names = 1, header = TRUE)
C.cont <- ca(cont)
print(C.cont)
i <- 0
for (val in C.cont) {
    i <- i + 1
    cat('\n\t', names(C.cont)[[i]], '\n')
    print(val)
}
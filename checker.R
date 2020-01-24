library(ca)

basic_test <- function(x) {
    cont <- read.csv(paste('data/', x, sep = ''), row.names = 1, header = TRUE)
    C.cont <- ca(cont)
    print(C.cont)
    i <- 0
    for (val in C.cont) {
        i <- i + 1
        cat('\n\t', names(C.cont)[[i]], '\n')
        print(val)
    }
    par(mfrow = c(2, 4))
    for (mt in c('symmetric', 'rowprincipal', 'colprincipal', 'symbiplot', 'rowgab', 'colgab', 'rowgreen', 'colgreen')) {
        plot(C.cont, map = mt, main = mt)
    }
    readline(prompt="Press [enter] to continue")
}

sup_test <- function(x) {
    cont <- read.csv(paste('data/', x, sep = ''), row.names = 1, header = TRUE)
    C.cont <- ca(cont, suprow=c(1))
    print(C.cont)
    i <- 0
    for (val in C.cont) {
        i <- i + 1
        cat('\n\t', names(C.cont)[[i]], '\n')
        print(val)
    }
}

lapply(list.files(path = 'data'), sup_test)

setGeneric("%+%", def = function(x, y)standardGeneric("%+%"))
setMethod("%+%", signature = c("character", "character"),
            definition = function(x, y)paste0(x, y))
setMethod("%+%", signature = c("character", "ANY"),
            definition = function(x, y)paste0(x, y))
setMethod("%+%", signature = c("ANY", "character"),
            definition = function(x, y)paste0(x, y))

home = "/home/chib/code/GLMPrototype/"
rCodeFolder = home %+% "r/"
dataFolder = home %+% "data/"



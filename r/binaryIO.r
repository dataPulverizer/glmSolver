# Binary IO For Numeric Data in R

# 1D Array IO
setGeneric("write1DArray", def = function(fileName, v, bitSize)standardGeneric("write1DArray"))
setMethod("write1DArray", signature = c("character", "numeric", "numeric"), 
          definition = function(fileName, v, bitSize){
            .file = file(fileName, "wb")
            on.exit(close(.file))
            writeBin(length(v), .file, 8)
            writeBin(v, .file, bitSize)
            return(0)
})
setGeneric("read1DArray", def = function(fileName, what, bitSize)standardGeneric("read1DArray"))
setMethod("read1DArray", signature = c("character", "character", "numeric"),
          definition = function(fileName, what, bitSize){
            .file = file(fileName, "rb")
            on.exit(close(.file))
            len = readBin(.file, "integer", 1, 8)
            v = readBin(.file, what, len, bitSize)
            return(v)
})

# 2D Array IO
setGeneric("write2DArray", def = function(fileName, m, bitSize)standardGeneric("write2DArray"))
setMethod("write2DArray", signature = c("character", "matrix", "numeric"),
           definition = function(fileName, m, bitSize){
             .file = file(fileName, "wb")
             on.exit(close(.file))
            writeBin(as.integer(c(nrow(m), ncol(m))), .file, 8)
            writeBin(as.vector(m), .file, bitSize)
            return(0)
})
setGeneric("read2DArray", def = function(fileName, what, bitSize)standardGeneric("read2DArray"))
setMethod("read2DArray", signature = c("character", "character", "numeric"),
          definition = function(fileName, what, bitSize){
            .file = file(fileName, "rb")
            on.exit(close(.file))
            .dim = readBin(.file, "integer", 2, 8); len = .dim[1] * .dim[2]
            m = readBin(.file, what, len, bitSize)
            dim(m) = .dim
            return(m)
})

# ND Array IO
setGeneric("writeNDArray", def = function(fileName, arr, bitSize)standardGeneric("writeNDArray"))
setMethod("writeNDArray", signature = c("character", "array", "numeric"),
          definition = function(fileName, arr, bitSize){
            .file = file(fileName, "wb")
            on.exit(close(.file))
            .dim = dim(arr)
            writeBin(as.integer(length(.dim)), .file, 8)
            writeBin(as.integer(.dim), .file, 8)
            writeBin(as.vector(arr), .file, bitSize)
            return(0)
})
setGeneric("readNDArray", def = function(fileName, what, bitSize)standardGeneric("readNDArray"))
setMethod("readNDArray", signature = c("character", "character", "numeric"),
          definition = function(fileName, what, bitSize){
            .file = file(fileName, "rb")
            on.exit(close(.file))
            .ldim = readBin(.file, "integer", 1, 8)
            .dim = readBin(.file, "integer", .ldim, 8); len = prod(.dim)
            arr = readBin(.file, what, len, bitSize)
            dim(arr) = .dim
            return(arr)
})


ioTest = function()
{
  # For 1D Array
  vec = runif(10)
  binFile = "1DRFile.bin"
  write1DArray(binFile, vec, 8)
  o1D = read1DArray(binFile, "numeric", 8)
  cat("Sum of Errors For 1D: ", sum((vec - o1D)^2), "\n")
  
  # For 2D Array
  mat = matrix(runif(12), ncol = 3)
  binFile = "2DRFile.bin"
  write2DArray(binFile, mat, 8)
  o2D = read2DArray(binFile, "numeric", 8)
  cat("Sum of Errors For 2D: ", sum((mat - o2D)^2), "\n")

  # For 3D Array
  tensor = array(runif(27), dim = c(3, 3, 3))
  writeNDArray("NDRFile.bin", tensor, 8)
  o3D = readNDArray("NDRFile.bin", "numeric", 8)
  cat("Sum of Errors For 3D: ", sum((tensor - o3D)^2), "\n")
  
  unlink("1DRFile.bin")
  unlink("2DRFile.bin")
  unlink("NDRFile.bin")
}

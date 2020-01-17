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
# Overload so that we don't need to do bitsize defaults at 8 bytes
# setGeneric("write2DArray", def = function(fileName, m)standardGeneric("write2DArray"))
setMethod("write2DArray", signature = c("character", "matrix"),
           definition = function(fileName, m){
             .file = file(fileName, "wb")
             on.exit(close(.file))
            writeBin(as.integer(c(nrow(m), ncol(m))), .file, 8)
            writeBin(as.vector(m), .file, 8)
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
# Overload so that we don't need to do bitsize defaults at 8 bytes
# setGeneric("read2DArray", def = function(fileName, what)standardGeneric("read2DArray"))
setMethod("read2DArray", signature = c("character", "character"),
          definition = function(fileName, what){
            .file = file(fileName, "rb")
            on.exit(close(.file))
            .dim = readBin(.file, "integer", 2, 8); len = .dim[1] * .dim[2]
            m = readBin(.file, what, len, 8)
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
#======================================================================#
# Block I/O
# Test function to print block indexes
printBlock = function(n = 100, nBlocks = 7)
{
  for(i in 0:(nBlocks - 1))
  {
    start = ((n * i) %/% nBlocks) + 1
    finish = ((n * (i + 1)) %/% nBlocks)
    mess = paste0("start: ", start, ", finish: ", finish,
      ", diff: ", finish - start, "\n")
    cat(mess)
  }
  return(invisible())
}
# Convert a matrix to a list of matrices
matrixToBlock = function(mat, nBlocks)
{
  n = nrow(mat)
  ret = vector(length = nBlocks, mode = "list")
  for(i in 0:(nBlocks - 1))
  {
    start = ((n * i) %/% nBlocks) + 1
    finish = ((n * (i + 1)) %/% nBlocks)
    ret[[i + 1]] = mat[start:finish, , drop = FALSE]
  }
  return(ret)
}
# Convert a vector to a list of vectors
vectorToBlock = function(vec, nBlocks)
{
  n = length(vec)
  ret = vector(length = nBlocks, mode = "list")
  for(i in 0:(nBlocks - 1))
  {
    start = ((n * i) %/% nBlocks) + 1
    finish = ((n * (i + 1)) %/% nBlocks)
    ret[[i + 1]] = vec[start:finish]
  }
  return(ret)
}
# Function to create random block matrix
createRandomBlockMatrix = function(nrows = 100, ncols = 10, nBlocks = 7)
{
  ret = vector(length = nBlocks, mode = "list")
  for(i in 0:(nBlocks - 1))
  {
    start = ((nrows * i) %/% nBlocks) + 1
    finish = ((nrows * (i + 1)) %/% nBlocks)
    .n = (finish - start) + 1
    ret[[i + 1]] = matrix(runif(.n*ncols), ncol = ncols)
  }
  return(ret)
}
# To make sure conversion to block matrix works
blockConvertTest = function(n = 100, nBlocks = 8)
{
  mat = matrix(runif(n*10), nrow = n)
  vec = runif(n)
  return(list("matrix" = matrixToBlock(mat, nBlocks),
              "vector" = vectorToBlock(vec, nBlocks)))
}
#=======================================================================#
#' @description This function writes a list of matrices representing 
#' blocks to disk in many files.
#' @param path the path to the folder where all the data blocks 
#'        should be written.
#' @param blockMatrix a list of matrices to be written to the folder
#' @param bitSize number of bits each element in the matrix or array 
#'
setGeneric("write2DBlock", def = function(path, blockMatrix, bitSize)standardGeneric("write2DBlock"))
setMethod("write2DBlock", signature = c("character", "list", "numeric"), 
  definition = function(path, blockMatrix, bitSize)
  {
    nFiles = length(blockMatrix)
    dir.create(path)
    for(i in 1:nFiles)
    {
      fileName = paste0(path, "/block_", i, ".bin")
      write2DArray(fileName, blockMatrix[[i]], bitSize)
    }
    return(invisible())
})
# Overload sets bitSize = 8
# setGeneric("write2DBlock", def = function(path, blockMatrix)standardGeneric("write2DBlock"))
setMethod("write2DBlock", signature = c("character", "list"), 
  definition = function(path, blockMatrix)
  {
    nFiles = length(blockMatrix)
    dir.create(path)
    for(i in 1:nFiles)
    {
      fileName = paste0(path, "/block_", i, ".bin")
      write2DArray(fileName, blockMatrix[[i]], 8)
    }
    return(invisible())
})
#' @description This function reads blocks of matrices from the 
#' files in a given folder.
#' @param path the path to the folder where all the data blocks
#' will be read from.
#' @param what character for the type of element in the matrix
#' blocks that will be read.
#' @param bitSize the number of bits of each element in the matrix
#' or array.
setGeneric("read2DBlock", def = function(path, what, bitSize)standardGeneric("read2DBlock"))
setMethod("read2DBlock", signature = c("character", "character", "numeric"),
  definition = function(path, what, bitSize)
  {
    files = list.files(path, full.names = TRUE)
    nFiles = length(files)
    ret = vector(length = nFiles, mode = "list")
    for(i in 1:nFiles)
    {
      ret[[i]] = read2DArray(files[i], what, bitSize)
    }
    return(ret)
})
# Overload sets bitSize = 8
# setGeneric("read2DBlock", def = function(path, what)standardGeneric("read2DBlock"))
setMethod("read2DBlock", signature = c("character", "character"),
  definition = function(path, what)
  {
    files = list.files(path, full.names = TRUE)
    nFiles = length(files)
    ret = vector(length = nFiles, mode = "list")
    for(i in 1:nFiles)
    {
      ret[[i]] = read2DArray(files[i], what, 8)
    }
    return(ret)
})
# Deletes block file by deleting the folder where the files are stored
deleteBlockFiles = function(path)
{
  unlink(path, recursive = TRUE)
}
# Function to test block IO
testBlockIO = function(nrows = 100, ncols = 10, nBlocks = 7, path = "tmp")
{
  blockMatrix = createRandomBlockMatrix(nrows, ncols, nBlocks)
  write2DBlock(path, blockMatrix)
  ret = read2DBlock(path, "numeric")
  isSame = sum(sapply(1:nBlocks, function(i){sum(abs(blockMatrix[[i]] - ret[[i]]))})) == 0
  cat(paste0("The read block matrices are the same as those written? ", isSame, "\n"))
  # Sys.sleep(5)
  deleteBlockFiles(path)
  return(invisible())
}

writeTestData = function()
{
  blockMatrix = createRandomBlockMatrix(nrows = 1000, ncols = 5, nBlocks = 8)
  write2DBlock("/home/chib/code/glmSolver/data/testData", blockMatrix)
  return(invisible())
}


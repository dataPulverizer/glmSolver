#=
  Binary IO
=#
"""
  Function to write 2D array to disk
"""
function write2DArray(arr::Array{T}, fileName::String) where {T <: AbstractFloat}
  
  # Opens Binary File
  io = open(fileName, "w")
  _size = size(arr)
  
  for i in _size
    write(io, Int64(i))
  end

  for el in arr
    write(io, el)
  end

  close(io)
  return 0

end

"""
  Function to read 2D array from disk
"""
function read2DArray(::Type{T}, fileName::String) where {T <: AbstractFloat}
  
  io = open(fileName, "r")

  _size = zeros(Int64, 2)
  
  for i in 1:2
    _size[i] = read(io, Int64)
  end
  
  arr = Array{T}(undef, tuple(_size...))
  
  for i in eachindex(arr)
    arr[i] = read(io, T)
  end
  return arr
end

"""
  Function to write an array to disk
"""
function writeNDArray(arr::Array{T}, fileName::String) where {T <: AbstractFloat}
  
  # Opens Binary File
  io = open(fileName, "w")
  _size = size(arr)
  
  write(io, Int64(length(_size)))
  
  for i in _size
    write(io, Int64(i))
  end

  for el in arr
    write(io, el)
  end

  close(io)
  return 0

end

"""
  Function to read an array from disk
"""
function readNDArray(::Type{T}, fileName::String) where {T <: AbstractFloat}
  io = open(fileName, "r")
  _len = read(io, Int64)
  _size = tuple(zeros(Int64, _len)...)
  
  for i in 1:_len
    _size[i] = read(io, Int64)
  end
  
  arr = Array{T}(undef, _size)
  
  for i in eachindex(arr)
    arr[i] = read(io, T)
  end
  
  return arr
end


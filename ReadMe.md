Just a 1-dim array addtion running on GPU device.
Tips: 
1. It looks CUDA doesn't support 2-dim array copy and retriving a[i][j].
2. Use dim3 to define the size of thread per block, and block size, dim3 threadPerBock(N), dim3 blockSize(M), M*N=array.size().
   dim3 threadPerBock(M,N) or dim3 blockSize(M,N).
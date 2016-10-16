1. cudaMemcpyAsync(), function is a non-blocking variant of cudaMemcpy() in which control is returned immediately to the host thread.
   example:
   /*
    cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, 0);
    kernel<<<grid, block>>>(a_d);
    cpuFunction();
    */
    The last argument to the cudaMemcpyAsync() function is the stream ID, which in this
    case uses the default stream, stream 0. 
    
2. Concurrent copy and execute
   /* cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, stream1);
    kernel<<<grid, block, 0, stream2>>>(otherData_d);
   */
   In this code, two streams are created and used in the data transfer and kernel executions
   as specified in the last arguments of the cudaMemcpyAsync call and the kernel's
   execution configuration.
   
3. Sequential copy and execute
   /* cudaMemcpy(a_d, a_h, N*sizeof(float), dir);
    kernel<<<N/nThreads, nThreads>>>(a_d);
   */
   
4. Staged concurrent copy and execute
   /* size=N*sizeof(float)/nStreams;
    for (i=0; i<nStreams; i++) {
        offset = i*N/nStreams;
        cudaMemcpyAsync(a_d+offset, a_h+offset, size, dir, stream[i]);
        kernel<<<N/(nThreads*nStreams), nThreads, 0,
        stream[i]>>>(a_d+offset);
    }
   */
   Staged concurrent copy and execute shows how the transfer and kernel execution can
   be broken up into nStreams stages. This approach permits some overlapping of the data
   transfer and execution.
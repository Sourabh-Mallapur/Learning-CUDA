# Learning-CUDA
The Graphics Processing Unit (GPU) provides much higher instruction throughput and memory bandwidth than the CPU within a similar price and power envelope. Many applications leverage these higher capabilities to run faster on the GPU than on the CPU. Other computing devices, like FPGAs, are also very energy efficient, but offer much less programming flexibility than GPUs.
This difference in capabilities between the GPU and the CPU exists because they are designed with different goals in mind. While the CPU is designed to excel at executing a sequence of operations, called a thread, as fast as possible and can execute a few tens of these threads in parallel, the GPU is designed to excel at executing thousands of them in parallel (amortizing the slower single-thread performance to achieve greater throughput).
The GPU is specialized for highly parallel computations and therefore designed such that more transistors are devoted to data processing rather than data caching and flow control. The schematic shows an example distribution of chip resources for a CPU versus a GPU.




Motivation:
•  GPU go brrr, more FLOPS please
• Why? Simulation & world-models (games, weather, proteins, robotics)
• Bigger models are smarter -> AGI (prevent wars, fix climate, cure cancer)
• GPUs are the backbone of modern deep learning
• classic software: sequential programs
• higher clock rate trend for CPU slowed in 2003: energy consumption & heat dissipation
• multi-core CPU came up
• developers had to learn multi-threading (deadlocks, races etc.)


The Rise of CUDA:
• CUDA is all about parallel programs (modern software)
• GPUs have (much) higher peak FLOPS than multi-core CPUs
• main principle: divide work among threads
• GPUs focus on execution throughput of massive number of threads
• programs with few threads perform poorly on GPUs
• CPU+GPU: sequential parts on CPU, numerical intensive parts on GPU
• CUDA: Compute Unified Device Architect
• GPGPU: Before CUDA tricks were used to compute with graphics APIs (OpenGL or Direct3D)
• GPU programming is now attractive for developers (thanks to massive availability)

General Terminologies - 
Host & Device: The host is often used to refer to the CPU, while device is used to refer to the GPU.
Thread: The smallest execution unit in a CUDA program.
Block: A set of CUDA threads sharing resources.
Grid: A set of blocks launched in one kernel.
Kernel: A large parallel loop, where each thread executes one iteration.

Kernels
CUDA C++ extends C++ by allowing the programmer to define C++ functions, called kernels, that, when called, are executed N times in parallel by N different CUDA threads, as opposed to only once like regular C++ functions.
A kernel is defined using the __global__ declaration specifier and the number of CUDA threads that execute that kernel for a given kernel call is specified using a new <<<...>>> execution configuration syntax. Each thread that executes the kernel is given a unique thread ID that is accessible within the kernel through built-in variables.








CUDA defines built-in 3D variables for threads and blocks. Threads are indexed using the built-in 3D variable threadIdx. Three-dimensional indexing provides a natural way to index elements in vectors, matrix, and volume and makes CUDA programming easier. Similarly, blocks are also indexed using the in-built 3D variable called blockIdx.
Here are a few noticeable points:
• CUDA architecture limits the numbers of threads per block (1024 threads per block limit).
• The dimension of the thread block is accessible within the kernel through the built-in blockDim variable.
• All threads within a block can be synchronized using an intrinsic function __syncthreads. With __syncthreads, all threads in the block must wait before anyone can proceed.
• The number of threads per block and the number of blocks per grid specified in the <<<…>>> syntax can be of type int or dim3. These triple angle brackets mark a call from host code to device code. It is also called a kernel launch.
The CUDA program for adding two matrices below shows multi-dimensional blockIdx and threadIdx and other variables like blockDim. In the example below, a 2D block is chosen for ease of indexing and each block has 256 threads with 16 each in x and y-direction. The total number of blocks are computed using the data size divided by the size of each block.
// Kernel - Adding two matrices MatA and MatB
__global__ void MatAdd(float MatA[N][N], float MatB[N][N],
float MatC[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        MatC[i][j] = MatA[i][j] + MatB[i][j];
}
 
int main()
{
    ...
    // Matrix addition kernel launch from host code
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x -1) / threadsPerBlock.x, (N+threadsPerBlock.y -1) / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(MatA, MatB, MatC);
    ...
}
Memory hierarchy
CUDA-capable GPUs have a memory hierarchy as depicted in Figure 4.

Figure 4. Memory hierarchy in GPUs.
The following memories are exposed by the GPU architecture:
• Registers—These are private to each thread, which means that registers assigned to a thread are not visible to other threads. The compiler makes decisions about register utilization.
• L1/Shared memory (SMEM)—Every SM has a fast, on-chip scratchpad memory that can be used as L1 cache and shared memory. All threads in a CUDA block can share shared memory, and all CUDA blocks running on a given SM can share the physical memory resource provided by the SM..
• Read-only memory—Each SM has an instruction cache, constant memory,  texture memory and RO cache, which is read-only to kernel code.
• L2 cache—The L2 cache is shared across all SMs, so every thread in every CUDA block can access this memory. The NVIDIA A100 GPU has increased the L2 cache size to 40 MB as compared to 6 MB in V100 GPUs.
• Global memory—This is the framebuffer size of the GPU and DRAM sitting in the GPU.
The NVIDIA CUDA compiler does a good job in optimizing memory resources but an expert CUDA developer can choose to use this memory hierarchy efficiently to optimize the CUDA programs as needed.

 


![image](https://github.com/Sourabh-Mallapur/Learning-CUDA/assets/106715050/663a7caf-fe20-4aa3-9810-346cec48f4e5)



## Resorces
Pasted from <https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/>
Pasted from <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#> 
https://youtu.be/9bBsvpg-Xlk?feature=shared
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

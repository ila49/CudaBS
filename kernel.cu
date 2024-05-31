// размер стека увеличен до 1'073'741'824 байт (1 ГБ) для конфигурации Release
// 1 073 741 824

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <ctime>
#include <stdio.h>

#define double float
#define BlockSide 14

std::clock_t start;
double duration;
const unsigned int Blocks = 176'326;
const unsigned int BlockSize = BlockSide * BlockSide;
const unsigned int TotalThreads = Blocks * BlockSize;

// const unsigned int *pBS = new const unsigned int(BlockSize);  // не работает

cudaError_t addWithCuda(double* Vx, double* Vy, double* Vz,
                        unsigned int size,
                        double Ro2, double Roo2);

__device__ double atomicAdd(double* address, double val);

//-----------------------------------------------------------------------------------------------------------------------------------------
__global__ void addKernel(double* Vx, double* Vy, double* Vz,
                          double Ro2, double Roo2,
                          double PIx4)
{
    //const int aaa = blockDim.x * blockDim.y;
//    int i = threadIdx.x;                                           // для одномерного блока
//    int i = threadIdx.x + 24 * threadIdx.y;                        // для двумерного блока
    int i = threadIdx.x + BlockSide * threadIdx.y + BlockSize * blockIdx.x;   // для нескольких двумерных блоков
    int i2 = threadIdx.x + BlockSide * threadIdx.y;

// суммарно 224 Б на поток, 43904 Б на блок (общая память блока - 49152 Б). Все нижевводимые
// переменные типа double должны уместиться в общей памяти блока
    __shared__ double X1[BlockSize], Y1[BlockSize], Z1[BlockSize], X2[BlockSize], Y2[BlockSize], Z2[BlockSize];

// координаты всех точек считаем на GPU, а не копируем из памяти хоста
    X1[i2] = 0.001 * i;
    Y1[i2] = 0.0;
    Z1[i2] = 0.0;
    X2[i2] = 1.0 + 0.002 * i;
    Y2[i2] = 0.0;
    Z2[i2] = 0.0;

    atomicAdd(&X2[i2], X1[i2]);
    Vx[i] = X2[i2];
//    Vx[i] = X1[i2] + X2[i2];
    atomicAdd(&Y2[i2], Y1[i2]);
    Vy[i] = Y2[i2];
//    Vy[i] = Y1[i2] + Y2[i2];
    atomicAdd(&Z2[i2], Z1[i2]);
    Vz[i] = Z2[i2];
//    Vz[i] = Z1[i2] + Z2[i2];
}
//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------
int main()
{
// общие сведения о графической карте
    cudaDeviceProp Device;
    cudaError_t err = cudaGetDeviceProperties(&Device, 0);
    printf("                     Device name: %s\n", Device.name);
    printf("             Total global memory: %d MB\n", (int)(Device.totalGlobalMem / 1024 / 1024));
    printf("         Shared memory per block: %zd\n", Device.sharedMemPerBlock);
    printf("             Registers per block: %d\n", Device.regsPerBlock);
    printf("                       Warp size: %d\n", Device.warpSize);
    printf("                    Memory pitch: %zd\n", Device.memPitch);
    printf("           Max threads per block: %d\n", Device.maxThreadsPerBlock);
    printf("          Max threads dimensions: x=%d, y=%d, z=%d\n", Device.maxThreadsDim[0], 
                                                                   Device.maxThreadsDim[1],
                                                                   Device.maxThreadsDim[2]);
    printf("                   Max grid size: x=%d, y=%d, z=%d\n", Device.maxGridSize[0],
                                                                   Device.maxGridSize[1],
                                                                   Device.maxGridSize[2]);
    printf("                      Clock rate: %d\n", Device.clockRate);
    printf("           Total constant memory: %zd\n", Device.totalConstMem);
    printf("              Compute capability: %d.%d\n", Device.major, Device.minor);
    printf("               Texture alignment: %zd\n", Device.textureAlignment);
    printf("                  Device overlap: %d\n", Device.deviceOverlap);
    printf("            Multiprocessor count: %d\n", Device.multiProcessorCount);
    printf("Kernel execution timeout enabled: %s\n", Device.kernelExecTimeoutEnabled ? "true" : "false");
    printf("     Concurrent copy and compute: %d\n", Device.asyncEngineCount);
//
    const unsigned int arraySize = Blocks * BlockSize; // 512; //1024;
    unsigned int arraySize1 = arraySize-1;
    double Vx[arraySize] = { 0.0 };
    double Vy[arraySize] = { 0.0 };
    double Vz[arraySize] = { 0.0 };
    double Ro2 = 0.01;
    double Roo2 = 0.001;

    start = std::clock();

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(Vx, Vy, Vz,
                                         arraySize,
                                         Ro2, Roo2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

//    printf("*pBS = %d\n", *pBS);
    printf("Vx[0] = %.12f\n", Vx[0]);
    printf("Vy[0] = %.12f\n", Vy[0]);
    printf("Vz[0] = %.12f\n", Vz[0]);
    printf("Vx[176326] = %.12f\n", Vx[176326]);
    printf("Vy[176326] = %.12f\n", Vy[176326]);
    printf("Vz[176326] = %.12f\n", Vz[176326]);
    printf("Vx[.] = %.12f\n", Vx[arraySize1]);
    printf("Vy[.] = %.12f\n", Vy[arraySize1]);
    printf("Vz[.] = %.12f\n", Vz[arraySize1]);
    printf("T = %.12f\n", duration);

    //delete(pBS);    // не работает

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(double* Vx, double* Vy, double* Vz,
                        unsigned int size, double Ro2, double Roo2)
{
    const double PIx4 = 4.0 * 3.141592653589793;
    double* dev_Vx = 0;
    double* dev_Vy = 0;
    double* dev_Vz = 0;
    cudaError_t cudaStatus;
    size_t Size = size * sizeof(double);
    dim3 threadsPerBlock(BlockSide, BlockSide); // для двумерного блока

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_Vx, Size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_Vy, Size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_Vz, Size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    // kernel <<< gridsize, blocksize, sharedmemory, streamid>>>  (args)
    // addKernel <<< 1, size >>> (dev_Vx, dev_Vy, dev_Vz,                 // для одномерного блока
    addKernel <<< Blocks, threadsPerBlock >>> (dev_Vx, dev_Vy, dev_Vz,    // для двумерных блоков
                                               Ro2, Roo2,
                                               PIx4);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(Vx, dev_Vx, Size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(Vy, dev_Vy, Size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(Vz, dev_Vz, Size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_Vx);
    cudaFree(dev_Vy);
    cudaFree(dev_Vz);

    return cudaStatus;
}
//-----------------------------------------------------------------------------------------------------------------------------------------
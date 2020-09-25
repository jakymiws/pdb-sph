#include <glew.h>
#include "FluidSimulator.h"

#include "common.h"

FluidSimulator::FluidSimulator(int n, int _cellSize, int _gridWidth)
{
    num_fluid_particles = n;

    gridWidth = _gridWidth;
    gridSize = gridWidth*gridWidth;

    cellSize = _cellSize;
    invCellSize = 1.0f/cellSize;

    x = new float[num_fluid_particles*4]; 
    p = new float[num_fluid_particles*4]; 
    v = new float[num_fluid_particles*4];

    RandomPositionStart();

    AllocCudaArrays();

    printf("New fluid simulator with %d particles, cellSize %d, invCellSize %f, gridWidth %d, gridSize %d\n", 
            num_fluid_particles, cellSize, invCellSize, gridWidth, gridSize);

    InitGL();
    printf("openGL initialized in fs with glVBO = %d\n", glVBO);
}

void FluidSimulator::AllocCudaArrays()
{
    cudaMalloc((void**)&dev_v, num_fluid_particles*4*sizeof(float));
    checkCUDAError("malloc dev_v failed");
    cudaMalloc((void**)&dev_p, num_fluid_particles*4*sizeof(float));
    checkCUDAError("malloc dev_p failed");
    // cudaMalloc((void**)&dev_x, num_fluid_particles*4*sizeof(float));
    // checkCUDAError("malloc dev_x failed");

    cudaMemcpy(dev_v, v, num_fluid_particles*4*sizeof(float), cudaMemcpyHostToDevice);
    checkCUDAError("memcpy v-->dev_v failed");

    // cudaMemcpy(dev_x, x, num_fluid_particles*4*sizeof(float), cudaMemcpyHostToDevice);
    // checkCUDAError("memcpy x-->dev_x failed");

    cudaMalloc((void**)&dev_cellIds, num_fluid_particles*sizeof(uint));
    checkCUDAError("malloc failed");
    cudaMalloc((void**)&dev_particleIds, num_fluid_particles*sizeof(uint));
    checkCUDAError("malloc failed");
    cudaMalloc((void**)&dev_cellStarts, gridSize*sizeof(uint));
    checkCUDAError("malloc failed");
}

void registerVBO_WithCUDA(const uint vbo, struct cudaGraphicsResource **cudaVBO_resource)
{
    cudaGraphicsGLRegisterBuffer(cudaVBO_resource, vbo, cudaGraphicsMapFlagsNone);
    checkCUDAError("register vbo failed");
}

void unregisterVBO_WithCUDA(struct cudaGraphicsResource *cudaVBO_resource)
{
    cudaGraphicsUnregisterResource(cudaVBO_resource);
    checkCUDAError("unregister buffer failed");
}

void *mapGL(struct cudaGraphicsResource **cuda_resource)
{
    void *ptr;
    cudaGraphicsMapResources(1, cuda_resource, 0);
    checkCUDAError("map error");
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, *cuda_resource);
    checkCUDAError("map error");
    return ptr;
}

void unmapGL(struct cudaGraphicsResource *cuda_resource)
{
    cudaGraphicsUnmapResources(1, &cuda_resource, 0);
    checkCUDAError("unmap error");
}

void FluidSimulator::InitGL()
{
    //create GL VBO and store in global var.
    //glGenVertexArrays(1, &glVAO);
    glGenBuffers(1, &glVBO);

    //glBindVertexArray(glVAO);
    glBindBuffer(GL_ARRAY_BUFFER, glVBO);
    glBufferData(GL_ARRAY_BUFFER, num_fluid_particles*4*sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    //glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    //glEnableVertexAttribArray(0);
    //register VBO With Cuda
    registerVBO_WithCUDA(glVBO, &cudaVBO_resource);

    unregisterVBO_WithCUDA(cudaVBO_resource);
    glBindBuffer(GL_ARRAY_BUFFER, glVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, num_fluid_particles*4*sizeof(float), x);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    registerVBO_WithCUDA(glVBO, &cudaVBO_resource);

}

void FluidSimulator::RandomPositionStart()
{
    for (int i = 0; i < num_fluid_particles; i++)
    {
        float xCoord = randomFloatRangef(0.0f, 0.5f);
        float yCoord = randomFloatRangef(0.0f, 0.5f);
        //float zCoord = randomFloatRange(0.0f, 0.5f);

        x[i*4] = xCoord; x[i*4+1] = yCoord; x[i*4+2] = 0.0f; x[i*4+3] = 0.0f;
        v[i*4] = 0.0f;   v[i*4+1] = 0.0f;   v[i*4+2] = 0.0f; v[i*4+3] = 0.0f;

    }
}

__global__ void computeSpatialHash(const int n, const float inv_cell_size, const int gridWidth, const float* ip, uint *oCellIds, uint *oParticleIds)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

    //hash position
    int cellId = ip[index*4]*inv_cell_size + ip[index*4+1]*inv_cell_size*gridWidth;
    
    oCellIds[index] = abs(cellId);
    oParticleIds[index] = index;
}

void thrustRadixSort(const int n, uint *ioCellIds, uint *ioParticleIds)
{
    thrust::sort_by_key(thrust::device_ptr<uint>(ioCellIds), 
                        thrust::device_ptr<uint>(ioCellIds + n), thrust::device_ptr<uint>(ioParticleIds));
}

__global__ void findCellsInArray(const int n, const uint* iCellIds, uint* cellStarts)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }
    //the first cell in the array starts at the 0th position.
    if (index == 0)
    {
        cellStarts[iCellIds[index]] = 0;
        return;
    }
    //if neighboring cellIds in the ihashes array (which is sorted by cellId) are different
    //then we have found the start of a new cell.
    if (iCellIds[index] != iCellIds[index-1])
    {
        cellStarts[iCellIds[index]] = index; 
    }
}

__global__ void explictEuler(int n, const float dt, float* ix, float* ov, float* op)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

    //float velo_damp = 0.99f;
    float g = -9.8f;
    float f_ext_x = 0.0f;
    float f_ext_y = g;
    float f_ext_z = 0.0f;

    ov[index*4]   = ov[index*4]   + dt*f_ext_x;
    ov[index*4+1] = ov[index*4+1] + dt*f_ext_y;
    ov[index*4+2] = ov[index*4+2] + dt*f_ext_z;

    ix[index*4]   = ix[index*4]   + dt*ov[index*4];
    ix[index*4+1] = ix[index*4+1] + dt*ov[index*4+1];
    ix[index*4+2] = ix[index*4+2] + dt*ov[index*4+2];

}

void FluidSimulator::stepSimulation(const float dt)
{
    //float dt = 0.0086f;
    //dev_x = glMap() somehow? I don't understand this bit.

    //predict using explicit euler
    // float *dev_v;
    // float *dev_p;
    // float *dev_x;

    float* dev_x = (float*) mapGL(&cudaVBO_resource);

    // cudaMalloc((void**)&dev_v, num_fluid_particles*4*sizeof(float));
    // checkCUDAError("malloc dev_v failed");
    // cudaMalloc((void**)&dev_p, num_fluid_particles*4*sizeof(float));
    // checkCUDAError("malloc dev_p failed");
    // cudaMalloc((void**)&dev_x, num_fluid_particles*4*sizeof(float));
    // checkCUDAError("malloc dev_x failed");

    // cudaMemcpy(dev_v, v, num_fluid_particles*4*sizeof(float), cudaMemcpyHostToDevice);
    // checkCUDAError("memcpy v-->dev_v failed");

    // cudaMemcpy(dev_x, x, num_fluid_particles*4*sizeof(float), cudaMemcpyHostToDevice);
    // checkCUDAError("memcpy x-->dev_x failed");

    int numThreads = 256;
    dim3 threadsPerBlock(numThreads);
    dim3 blocksPerGrid((num_fluid_particles+numThreads-1)/numThreads);

    //call explict euler per particle. Modifies dev_v and dev_p
    explictEuler<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, dt, dev_x, dev_v, dev_p);
    checkCUDAError("explicit euler failed");

    //cudaDeviceSynchronize();
    
    //rehash each step
    // uint *dev_cellIds;
    // uint *dev_particleIds;
    // uint *dev_cellStarts;

    // cudaMalloc((void**)&dev_cellIds, num_fluid_particles*sizeof(uint));
    // checkCUDAError("malloc failed");
    // cudaMalloc((void**)&dev_particleIds, num_fluid_particles*sizeof(uint));
    // checkCUDAError("malloc failed");
    // cudaMalloc((void**)&dev_cellStarts, gridSize*sizeof(uint));//should actually be the number of cells - will need to calc this
    // checkCUDAError("malloc failed");

    //compute spatial hashes per particle. Modifies dev_CellIds and dev_particleIds
    computeSpatialHash<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, invCellSize, gridWidth, dev_p, dev_cellIds, dev_particleIds);
    checkCUDAError("computeSpatialHash failed");

    // cudaDeviceSynchronize();
    // //sort by cellId
    thrustRadixSort(num_fluid_particles, dev_cellIds, dev_particleIds);
    checkCUDAError("thrust error");

    // cudaDeviceSynchronize();
    // //get starting index of each cellId in the cellIds and particleIds parallel arrays and store in dev_cellStarts.
    findCellsInArray<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, dev_cellIds, dev_cellStarts);
    checkCUDAError("findCellsInArray failed");
    // cudaDeviceSynchronize();
    uint* cellIds = new uint[num_fluid_particles];
    uint* particleIds = new uint[num_fluid_particles];
    uint* cellStarts = new uint[gridSize];
    
    cudaMemcpy(cellIds, dev_cellIds, num_fluid_particles*sizeof(uint), cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy failed");
    cudaMemcpy(particleIds, dev_particleIds, num_fluid_particles*sizeof(uint), cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy failed");
    cudaMemcpy(cellStarts, dev_cellStarts, gridSize*sizeof(uint), cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy failed");

   // cudaDeviceSynchronize();

    // cudaMemcpy(p, dev_p, num_fluid_particles*4*sizeof(float), cudaMemcpyDeviceToHost);
    // checkCUDAError("memcpy failed");
    // cudaMemcpy(v, dev_v, num_fluid_particles*4*sizeof(float), cudaMemcpyDeviceToHost);
    // checkCUDAError("memcpy failed");

    // for (int i = 0; i < num_fluid_particles; i++)
    // {
    //     printf("p[%d] = (%f,%f,%f,%f)\n", i, p[i*4], p[i*4+1], p[i*4+2], p[i*4+3]);
    // }
    // for (int i = 0; i < num_fluid_particles; i++)
    // {
    //     printf("v[%d] = (%f,%f,%f,%f)\n", i, v[i*4], v[i*4+1], v[i*4+2], v[i*4+3]);
    // }

    // for (int i = 0; i < num_fluid_particles; i++)
    // {
    //     printf("gs = %d\n", gridSize);
    //     printf("cellIds[%d] = %d\n", i, cellIds[i]);
    //     printf("particleIds[%d] = %d\n", i, particleIds[i]);
    //     printf("cellStarts[%d] = %d\n", cellIds[i], cellStarts[cellIds[i]]);

    // }
    // for (int i = 0; i < gridSize; i++)
    // {
    //     printf("cellStarts[%d] = %d\n", i, cellStarts[i]);
    // }

    //solve


    unmapGL(cudaVBO_resource);
    

}

float FluidSimulator::randomFloatf()
{
    return (rand() / (float)RAND_MAX);
}

float FluidSimulator::randomFloatRangef(float a, float b)
{
    float f = randomFloatf();
    float diff = b-a;
    float r = f*diff;
    return a+r;
}

uint FluidSimulator::getVBO(){ return glVBO; }
int FluidSimulator::getNumFluidParticles(){ return num_fluid_particles; }

void FluidSimulator::cleanUpSimulation()
{
    cudaFree(dev_p);
    //cudaFree(dev_x);
    cudaFree(dev_v);
    cudaFree(dev_cellIds);
    cudaFree(dev_particleIds);
    cudaFree(dev_cellStarts);
}
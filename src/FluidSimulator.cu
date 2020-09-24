#include "FluidSimulator.h"

#include "common.h"

FluidSimulator::FluidSimulator(int n)
{
    num_fluid_particles = n;

    x = new float4[num_fluid_particles]; 
    p = new float4[num_fluid_particles]; 
    v = new float4[num_fluid_particles];

    RandomPositionStart();

    printf("New fluid simulator with %d particles\n", num_fluid_particles);
}

void FluidSimulator::RandomPositionStart()
{
    for (int i = 0; i < num_fluid_particles; i++)
    {
        float xCoord = randomFloatRangef(0.0f, 0.5f);
        float yCoord = randomFloatRangef(0.0f, 0.5f);
        //float zCoord = randomFloatRange(0.0f, 0.5f);

        x[i] = make_float4(xCoord, yCoord, 0.0f, 0.0f);
        v[i] = make_float4(0.0f,0.0f,0.0f,0.0f);
    }
}

__global__ void computeSpatialHash(const int n, const float inv_cell_size, const int gridWidth, const float4* ip, uint *oCellIds, uint *oParticleIds)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }
    //(int)pos.x*inv_cell_size + ((int)pos.y*inv_cell_size)*gridWidth;

    float4 p_i = ip[index];
    //hash position
    int cellId = p_i.x*inv_cell_size + p_i.y*inv_cell_size*gridWidth;
    
    //ohashes[index] = make_uint2(cellId, index);
    oCellIds[index] = cellId;
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
        cellStarts[ihashes[0].x] = 0;
        return;
    }
    //if neighboring cellIds in the ihashes array (which is sorted by cellId) are different
    //then we have found the start of a new cell.
    if (iCellIds[index] != iCellIds[index-1])
    {
        cellStarts[iCellIds[index]] = index; 
    }
}

__global__ void explictEuler(int n, const float dt, const float4* ix, float4 *ov, float4* op)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

    float velo_damp = 0.99f;
    float g = -9.8f;
    float4 f_ext = make_float4(0.0f, g, 0.0f, 0.0f);

    float4 v_i = ov[index];
    float4 x_i = ix[index];

    v_i = make_float4(v_i.x + dt*f_ext.x, 
                         v_i.y + dt*f_ext.y, 
                         v_i.z + dt*f_ext.z, 
                         v_i.w + dt*f_ext.w);

    v_i = make_float4(v_i.x*velo_damp, v_i.y*velo_damp, v_i.z*velo_damp, v_i.w*velo_damp);
    ov[index] = v_i;

    op[index] = make_float4(x_i.x + dt*v_i.x,
                            x_i.y + dt*v_i.y, 
                            x_i.z + dt*v_i.z, 
                            x_i.w + dt*v_i.w);

}

void FluidSimulator::predictSimStepGPU()
{
    float dt = 0.0086f;

    spatialHashMap.clear();

    float4 *dev_v;
    float4 *dev_p;
    float4 *dev_x;

    uint2* particleHashes; //cudamalloc


    cudaMalloc((void**)&dev_v, num_fluid_particles*sizeof(float4));
    checkCUDAError("malloc dev_v failed");
    cudaMalloc((void**)&dev_p, num_fluid_particles*sizeof(float4));
    checkCUDAError("malloc dev_p failed");
    cudaMalloc((void**)&dev_x, num_fluid_particles*sizeof(float4));
    checkCUDAError("malloc dev_x failed");

    //copy velocities and positions (x) to device.
    cudaMemcpy(dev_v, v, num_fluid_particles*sizeof(float4), cudaMemcpyHostToDevice);
    checkCUDAError("memcpy v-->dev_v failed");

    cudaMemcpy(dev_x, x, num_fluid_particles*sizeof(float4), cudaMemcpyHostToDevice);
    checkCUDAError("memcpy x-->dev_x failed");

    int numThreads = 256;
    dim3 threadsPerBlock(numThreads);
    dim3 blocksPerGrid((num_fluid_particles+numThreads-1)/numThreads);

    //call explict euler. Modifies dev_v and dev_p
    explictEuler<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, dt, dev_x, dev_v, dev_p);
    checkCUDAError("explicit euler failed");

//__global__ void computeSpatialHash(const int n, const float inv_cell_size, const int gridWidth, const float4* ip, uint *oCellIds, uint *oParticleIds)
    computeSpatialHash<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, 1.0f, 2, dev_p, );


    cudaMemcpy(p, dev_p, num_fluid_particles*sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, dev_v, num_fluid_particles*sizeof(float4), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < num_fluid_particles; i++)
    // {
    //     printf("p[%d] = (%f,%f,%f,%f)\n", i, p[i].x, p[i].y, p[i].z, p[i].w);
    // }
    // for (int i = 0; i < num_fluid_particles; i++)
    // {
    //     printf("v[%d] = (%f,%f,%f,%f)\n", i, v[i].x, v[i].y, v[i].z, v[i].w);

    // }

    

    //computeSpatialHash
    //fast radix sort
    //findCellStart

    //also a good idea to sort the position/velocity arrays by particleId? GridId somehow?
}

void test()
{
    // int n = 10;

    // int *dev_odata;
    // int *odata = new int[n];
    // int *idata = new int[n];

    // for (int i = 0; i < n; i++)
    // {
    //     idata[i] = 0;
    //     odata[i] = 0;
    //     printf("pre odata[%d] = %d\n", i , odata[i]);
    // }


    // cudaMalloc((void**)&dev_odata, n*sizeof(int));
    // checkCUDAError("malloc failed");

    // cudaMemcpy(dev_odata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
    // checkCUDAError("memcpy failed");

    // int numThreads = 256;
    // dim3 threadsPerBlock(numThreads);
    // dim3 blocksPerGrid((n+numThreads-1)/numThreads);
    
    // //explictEuler<<<blocksPerGrid, threadsPerBlock>>>(n, dev_odata);
    // checkCUDAError("explicitEuler failed");

    // cudaMemcpy(odata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
    // checkCUDAError("memcpy failed");
    
    // for (int i = 0; i < n; i++)
    // {
    //     printf("post cuda odata[%d] = %d\n", i , odata[i]);
    // }

    // cudaFree(dev_odata);
}

int FluidSimulator::spatial_hash(const float x, const float y, const float z)
{
    return (int)x*inv_cell_size + ((int)y*inv_cell_size)*gridWidth;
}

void FluidSimulator::hashParticlePositions()
{
    //generate grid from scratch. Use a loose grid where each particle can only be in one cell and store these 
    //cell ids in an int array parallel to the state var ones.p
    for (int i = 0; i < num_fluid_particles; i++)
    {
        //I think it should be p[i] and not x[i] since I'm only calling this at the moment after the predict step.
        int gridCell = spatial_hash(p[i].x, p[i].y, p[i].z);
        spatialHashMap[gridCell].push_front(i);//push back the index of the xcoord.
    }
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
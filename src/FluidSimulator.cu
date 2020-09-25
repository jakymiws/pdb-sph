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

    h = 0.1f;
    rho0 = 6378.0f;
    epsR = 600.0f;

    maxIterations = 4;

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

    cudaMalloc((void**)&dev_density, num_fluid_particles*sizeof(float));
    checkCUDAError("malloc dev_p failed");
    cudaMalloc((void**)&dev_lambda, num_fluid_particles*sizeof(float));
    checkCUDAError("malloc dev_p failed");
    

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
    cudaMalloc((void**)&dev_cellEnds, gridSize*sizeof(uint));
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

__device__ int getCellNeighbors(const int cell, const int gridWidth, int *oCellNeighbors)
{
    int num_neighbors_found = 0;
    //neighboringCells.push_back(cell);  
    oCellNeighbors[num_neighbors_found] = cell;
    num_neighbors_found++;
    if (((cell + 1) % gridWidth) > (cell % gridWidth) && (cell+1) < gridWidth*gridWidth)
    {
        //neighboringCells.push_back(cell+1);
        oCellNeighbors[num_neighbors_found] = cell+1;
        num_neighbors_found++;
        //printf("ocn[%d] = %d\n", num_neighbors_found, cell+1);
        if (cell+1+gridWidth < gridWidth*gridWidth)
        {
            oCellNeighbors[num_neighbors_found] = cell+1+gridWidth;
            num_neighbors_found++;
        }
        if (cell+1-gridWidth > 0)
        {
            oCellNeighbors[num_neighbors_found] = cell+1-gridWidth;
            num_neighbors_found++;

        }
        //printf("ocn[%d] = %d\n", num_neighbors_found, cell+1-gridWidth);

        //neighboringCells.push_back(cell+1+gridWidth); neighboringCells.push_back(cell+1-gridWidth);
       // num_neighbors_found +=3;
        //printf("nnf = %d\n", num_neighbors_found);

    }
    
    if (((cell - 1) % gridWidth) < (cell % gridWidth) && (cell-1)>=0)
    {
        //neighboringCells.push_back(cell-1); 
        oCellNeighbors[num_neighbors_found] = cell-1;
        num_neighbors_found++;
        //printf("ocn[%d] = %d\n", num_neighbors_found, cell-1);
        if (cell-1+gridWidth < gridWidth*gridWidth)
        {
            oCellNeighbors[num_neighbors_found] = cell-1+gridWidth;
            num_neighbors_found++;
        }

        if (cell-1-gridWidth > 0)
        {
            oCellNeighbors[num_neighbors_found] = cell-1-gridWidth;
            num_neighbors_found++;
        }
        //printf("ocn[%d] = %d\n", num_neighbors_found, cell-1+gridWidth);
        //printf("ocn[%d] = %d\n", num_neighbors_found, cell-1-gridWidth);

        //neighboringCells.push_back(cell-1+gridWidth); neighboringCells.push_back(cell-1-gridWidth);
        //printf("nnf = %d\n", num_neighbors_found);

    }
    
    //neighboringCells.push_back(cell+gridWidth);
    //neighboringCells.push_back(cell-gridWidth);
    if (cell+gridWidth < gridWidth*gridWidth)
    {  
        oCellNeighbors[num_neighbors_found] = cell+gridWidth;
        num_neighbors_found++;
    } 

    if (cell-gridWidth > 0)
    {
        oCellNeighbors[num_neighbors_found] = cell-gridWidth;
        num_neighbors_found++;
    }
    //printf("nnf = %d\n", num_neighbors_found);
    return num_neighbors_found;
}



__global__ void computeSpatialHash(const int n, const float inv_cell_size, const int gridWidth, const float* ip, uint *oCellIds, uint *oParticleIds)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

    //hash position
    int cellId = (int)ip[index*4]*inv_cell_size + ((int)ip[index*4+1]*inv_cell_size)*gridWidth;
    
    oCellIds[index] = abs(cellId);
    oParticleIds[index] = index;
}

void thrustRadixSort(const int n, uint *ioCellIds, uint *ioParticleIds)
{
    thrust::sort_by_key(thrust::device_ptr<uint>(ioCellIds), 
                        thrust::device_ptr<uint>(ioCellIds + n), thrust::device_ptr<uint>(ioParticleIds));
}

__global__ void findCellsInArray(const int n, const int gridWidth, const uint* iCellIds, uint* cellStarts, uint* cellEnds)
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
        
    } else {
        if (iCellIds[index] != iCellIds[index-1])
        {
            cellStarts[iCellIds[index]] = index; 
        }
    }

    if (index+1 < n)
    {
        if(iCellIds[index] != iCellIds[index+1])
        {
            cellEnds[iCellIds[index]] = index;
        }
    } else {
        cellEnds[iCellIds[index]] = index;
    }
}

__global__ void explictEuler(int n, const float dt, float* ix, float* ov, float* op)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

    float velo_damp = 0.99f;
    float g = -9.8f;
    if (op[index*4+1] <= 0)
    {
        g = 0.0f;
    } 
    float f_ext_x = 0.0f;
    float f_ext_y = g;
    float f_ext_z = 0.0f;

    ov[index*4]   = ov[index*4]   + dt*f_ext_x;
    ov[index*4+1] = ov[index*4+1] + dt*f_ext_y;
    ov[index*4+2] = ov[index*4+2] + dt*f_ext_z;

    ov[index*4] *= velo_damp;
    ov[index*4+1] *= velo_damp;
    ov[index*4+2] *= velo_damp;

    op[index*4]   = ix[index*4]   + dt*ov[index*4];
    op[index*4+1] = ix[index*4+1] + dt*ov[index*4+1];
    op[index*4+2] = ix[index*4+2] + dt*ov[index*4+2];

}

__global__ void computeDensity(int n, float h, int gridWidth, const float* ip, const uint* cellIds, const uint* cellStarts, const uint* cellEnds, const uint* particleIds, float* odensity)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

    float h8 = powf(h, 8);
    float h2 = h*h;
    float _pi = 3.141592f;
    int particleId = particleIds[index];
    int cell = cellIds[index];

    int neighboringCells[9];
    for (int i = 0; i < 9; i++)
    {
        neighboringCells[i] = 0;
    }

    int num_neighbors_found = getCellNeighbors(cell, gridWidth, neighboringCells);

    float rho = 0.0f;
    for (int k = 0; k < num_neighbors_found; k++)
    {
        uint nCell = neighboringCells[k];
        uint start = cellStarts[nCell];
        uint end = cellEnds[nCell];

        if (start == end)
        {
            continue;
        }

        //loop through neighbors
            
        for (int i = start; i < end+1; i++)
        {
            int cellId_i = cellIds[i];
            int particleId_i = particleIds[i];
            if (cellId_i != nCell)
            {
                //we've reached the end of the cell
                break;
            }

            float rx = ip[particleId*4] - ip[particleId_i*4];
            float ry = ip[particleId*4+1] - ip[particleId_i*4+1];
            float rz = ip[particleId*4+2] - ip[particleId_i*4+2];

            float rd2 = rx*rx + ry*ry + rz*rz;
            if (rd2 < h2)
            {
                float W = (4.0f)/(_pi*h8)*powf((h2 - rd2),3.0f);
                rho += W;
            }
        }
    }

    odensity[particleId] = rho;

}

__global__ void computeLambda(int n, float h, int gridWidth, float rho0, float epsR, const float* ip, const uint* cellIds, const uint* cellStarts, const uint* cellEnds, const uint* particleIds, const float* idensity, float* olambda)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

    float h8 = powf(h, 8);
    float h2 = h*h;
    float _pi = 3.141592f;
    int particleId = particleIds[index];
    int cell = cellIds[index];

    float C_i = (idensity[particleId]/rho0) - 1.0f;

    int neighboringCells[9];
    for (int i = 0; i < 9; i++)
    {
        neighboringCells[i] = 0;
    }

    int num_neighbors_found = getCellNeighbors(cell, gridWidth, neighboringCells);

    float sum_grad_C_i = 0.0f;
    for (int k = 0; k < num_neighbors_found; k++)
    {
        uint nCell = neighboringCells[k];
        uint start = cellStarts[nCell];
        uint end = cellEnds[nCell];

        if (start == end)
        {
            continue;//cell is empty
        }

        //loop through neighbors
        for (int i = start; i < end+1; i++)
        {
            int cellId_i = cellIds[i];
            int particleId_i = particleIds[i];
            if (cellId_i != nCell)
            {
                //we've reached the end of the cell
                break;
            }

            float rx = ip[particleId*4] - ip[particleId_i*4];
            float ry = ip[particleId*4+1] - ip[particleId_i*4+1];
            float rz = ip[particleId*4+2] - ip[particleId_i*4+2];

            float rd2 = rx*rx + ry*ry + rz*rz;
            float rd = sqrtf(rd2);
            if (rd2 < h2)
            {
                float gradW_x = -(45.0f/((float)_pi*h2*h2*h2))*((h-rd)*(h-rd)*rx);
                float gradW_y = -(45.0f/((float)_pi*h2*h2*h2))*((h-rd)*(h-rd)*ry);
                float gradW_z = -(45.0f/((float)_pi*h2*h2*h2))*((h-rd)*(h-rd)*rz);

                gradW_x /= rho0;
                gradW_y /= rho0;
                gradW_z /= rho0;

                sum_grad_C_i += gradW_x*gradW_x + gradW_y*gradW_y + gradW_z*gradW_z;
            }
        }
    }
    
    olambda[particleId] = -C_i/(sum_grad_C_i+epsR);

}

__global__ void projectDensityConstraint(int n, float h, int gridWidth, float rho0, float* op, const uint* cellIds, const uint* cellStarts, const uint* cellEnds, const uint* particleIds, const float* idensity, const float* ilambda)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

    float h2 = h*h;
    float _pi = 3.141592f;
    int particleId = particleIds[index];
    int cell = cellIds[index];

    int neighboringCells[9];

    for (int i = 0; i < 9; i++)
    {
        neighboringCells[i] = 0;
    }

    int num_neighbors_found = getCellNeighbors(cell, gridWidth, neighboringCells);

    float constraint_sum_x = 0.0f; float constraint_sum_y = 0.0f; float constraint_sum_z = 0.0f;  
    for (int k = 0; k < num_neighbors_found; k++)
    {
        uint nCell = neighboringCells[k];
        uint start = cellStarts[nCell];
        uint end = cellEnds[nCell];

        if (start == end)
        {
            continue;
        }

        //loop through neighbors
            
        for (int i = start; i < end+1; i++)
        {
            int cellId_i = cellIds[i];
            int particleId_i = particleIds[i];
            if (cellId_i != nCell)
            {
                //we've reached the end of the cell
                break;
            }

            float rx = op[particleId*4] - op[particleId_i*4];
            float ry = op[particleId*4+1] - op[particleId_i*4+1];
            float rz = op[particleId*4+2] - op[particleId_i*4+2];

            float rd2 = rx*rx + ry*ry + rz*rz;
            float rd = sqrtf(rd2);
            if (rd2 < h2)
            {
                float W_s = (15.0f/(_pi*h2*h2*h2))*powf((h-rd),3.0f);
                float W_dq = (15.0f/(_pi*h2*h2*h2))*powf((0.2f),3.0f);
                float s_corr = 0.1*h*powf(W_s/W_dq, 4); 
            
                float gradW_x = -(45.0f/((float)M_PI*h2*h2*h2))*((h-rd)*(h-rd)*rx);
                float gradW_y = -(45.0f/((float)M_PI*h2*h2*h2))*((h-rd)*(h-rd)*ry);
                float gradW_z = -(45.0f/((float)M_PI*h2*h2*h2))*((h-rd)*(h-rd)*rz);
                
                constraint_sum_x += (ilambda[particleId] + ilambda[particleId_i] + s_corr) * gradW_x;
                constraint_sum_y += (ilambda[particleId] + ilambda[particleId_i] + s_corr) * gradW_y;
                constraint_sum_z += (ilambda[particleId] + ilambda[particleId_i] + s_corr) * gradW_z;
            }
        }
    }

    op[particleId*4] += constraint_sum_x/rho0;
    op[particleId*4+1] += constraint_sum_y/rho0;
    op[particleId*4+2] += constraint_sum_z/rho0;

}

__global__ void updatePositions(int n, float dt, const float *ip, float *ov, float *ox)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

    ov[index*4] = (ip[index*4]-ox[index*4])/dt;
    ov[index*4+1] = (ip[index*4+1]-ox[index*4+1])/dt;
    ov[index*4+2] = (ip[index*4+2]-ox[index*4+2])/dt;
    if (ip[index*4+1] <= 0)
    {
        ov[index*4] *= -0.3f;
        ov[index*4+1] *= -0.3f;
        ov[index*4+2] *= -0.3f;
    }

    ox[index*4] = ip[index*4];
    ox[index*4+1] = ip[index*4+1];
    ox[index*4+2] = ip[index*4+2];

}


void FluidSimulator::stepSimulation(const float dt)
{
    float* dev_x = (float*) mapGL(&cudaVBO_resource);

    int numThreads = 256;
    dim3 threadsPerBlock(numThreads);
    dim3 blocksPerGrid((num_fluid_particles+numThreads-1)/numThreads);

    //call explict euler per particle. Modifies dev_v and dev_p
    explictEuler<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, dt, dev_x, dev_v, dev_p);
    checkCUDAError("explicit euler failed");

    //compute spatial hashes per particle. Modifies dev_CellIds and dev_particleIds
    computeSpatialHash<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, invCellSize, gridWidth, dev_p, dev_cellIds, dev_particleIds);
    checkCUDAError("computeSpatialHash failed");

    //sort by cellId
    thrustRadixSort(num_fluid_particles, dev_cellIds, dev_particleIds);
    checkCUDAError("thrust error");

    //get starting index of each cellId in the cellIds and particleIds parallel arrays and store in dev_cellStarts.
    findCellsInArray<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, gridWidth, dev_cellIds, dev_cellStarts, dev_cellEnds);
    checkCUDAError("findCellsInArray failed");

    // uint* cellIds = new uint[num_fluid_particles];
    // uint* particleIds = new uint[num_fluid_particles];
    // uint* cellStarts = new uint[gridSize];
    // uint* cellEnds = new uint[gridSize];

    // cudaMemcpy(cellIds, dev_cellIds, num_fluid_particles*sizeof(uint), cudaMemcpyDeviceToHost);
    // checkCUDAError("memcpy failed");
    // cudaMemcpy(particleIds, dev_particleIds, num_fluid_particles*sizeof(uint), cudaMemcpyDeviceToHost);
    // checkCUDAError("memcpy failed");
    // cudaMemcpy(cellStarts, dev_cellStarts, gridSize*sizeof(uint), cudaMemcpyDeviceToHost);
    // checkCUDAError("memcpy failed");
    // cudaMemcpy(cellEnds, dev_cellEnds, gridSize*sizeof(uint), cudaMemcpyDeviceToHost);
    // checkCUDAError("memcpy failed");
    
    int num_iterations = 0;
    while (num_iterations < maxIterations)
    {
        computeDensity<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, h, gridWidth, dev_p, dev_cellIds, dev_cellStarts, dev_cellEnds, dev_particleIds, dev_density);
        checkCUDAError("computeDensity failed");

        computeLambda<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, h, gridWidth, rho0, epsR, dev_p, dev_cellIds, dev_cellStarts, dev_cellEnds, dev_particleIds, dev_density, dev_lambda);
        checkCUDAError("computeLambda failed");

        projectDensityConstraint<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, h, gridWidth, rho0, dev_p, dev_cellIds, dev_cellStarts, dev_cellEnds, dev_particleIds, dev_density, dev_lambda);
        checkCUDAError("projectDensityConstraint failed");

        cudaDeviceSynchronize();
        num_iterations++;
    }

    // float *hrho = new float[num_fluid_particles];
    // cudaMemcpy(hrho, dev_density, num_fluid_particles*sizeof(float), cudaMemcpyDeviceToHost);
    // checkCUDAError("memcpy failed");

    // for (int i = 0; i < num_fluid_particles; i++)
    // {
    //     printf("rho[%d] = %f\n", i, hrho[i]);
    // }
    

    updatePositions<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, dt, dev_p, dev_v, dev_x);
    checkCUDAError("updatePositions failed");
    
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
    cudaFree(dev_cellEnds);

    cudaFree(dev_density);
    cudaFree(dev_lambda);
}
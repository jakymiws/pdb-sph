#include <glew.h>
#include "FluidSimulator.h"

#include "common.h"

#include <glm/glm.hpp>

thrust::device_ptr<uint> dev_thrust_cellIds;
thrust::device_ptr<uint> dev_thrust_particleIds;

PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

FluidSimulator::FluidSimulator(int n, float _cellSize, int _gridWidth)
{
    num_fluid_particles = n;

    gridWidth = _gridWidth;
    gridWidth2 = gridWidth*gridWidth;
    gridSize = gridWidth*gridWidth*gridWidth;

    cellSize = _cellSize;
    invCellSize = 1.0f/cellSize;

    x = new float[num_fluid_particles*3]; 
    p = new glm::vec3[num_fluid_particles]; 
    v = new glm::vec3[num_fluid_particles];

    h = 0.1f;
    rho0 = 6378.0f;
    invRho0 = 1.0f/rho0;
    epsR = 600.0f;

    maxIterations = 3;

    RandomPositionStart();

    AllocCudaArrays();

    printf("New fluid simulator with %d particles, cellSize %f, invCellSize %f, gridWidth %d, gridSize %d\n", 
            num_fluid_particles, cellSize, invCellSize, gridWidth, gridSize);

    InitGL();
    printf("openGL initialized in fs with glVBO = %d\n", glVBO);
}

void FluidSimulator::AllocCudaArrays()
{
    cudaMalloc((void**)&dev_v, num_fluid_particles*sizeof(glm::vec3));
    checkCUDAError("malloc dev_v failed");
    cudaMalloc((void**)&dev_p, num_fluid_particles*sizeof(glm::vec3));
    checkCUDAError("malloc dev_p failed");
    cudaMalloc((void**)&dev_p2, num_fluid_particles*sizeof(glm::vec3));
    checkCUDAError("malloc dev_p failed");
    cudaMalloc((void**)&dev_p_lastFrame, num_fluid_particles*sizeof(glm::vec3));
    checkCUDAError("malloc dev_p failed");
    cudaMalloc((void**)&dev_sorted_p, num_fluid_particles*sizeof(glm::vec3));
    checkCUDAError("malloc dev_p failed");
    cudaMalloc((void**)&dev_sorted_v, num_fluid_particles*sizeof(glm::vec3));
    checkCUDAError("malloc dev_p failed");
    cudaMalloc((void**)&dev_p_lastFrame_sorted, num_fluid_particles*sizeof(glm::vec3));
    checkCUDAError("malloc dev_p failed");
    // cudaMalloc((void**)&dev_x, num_fluid_particles*4*sizeof(float));
    // checkCUDAError("malloc dev_x failed");

    // cudaMalloc((void**)&dev_density, num_fluid_particles*sizeof(float));
    // checkCUDAError("malloc dev_p failed");
    cudaMalloc((void**)&dev_lambda, num_fluid_particles*sizeof(float));
    checkCUDAError("malloc dev_lamba failed");
    
    cudaMemcpy(dev_v, v, num_fluid_particles*sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAError("memcpy v-->dev_v failed");

    // cudaMemcpy(dev_p, x, num_fluid_particles*sizeof(glm::vec3), cudaMemcpyHostToDevice);
    // checkCUDAError("memcpy x-->dev_p failed");

    cudaMalloc((void**)&dev_cellIds, num_fluid_particles*sizeof(uint));
    checkCUDAError("malloc failed");
    cudaMalloc((void**)&dev_particleIds, num_fluid_particles*sizeof(uint));
    checkCUDAError("malloc failed");
    cudaMalloc((void**)&dev_cellStarts, gridSize*sizeof(uint));
    checkCUDAError("malloc failed");
    cudaMalloc((void**)&dev_cellEnds, gridSize*sizeof(uint));
    checkCUDAError("malloc failed");

    dev_thrust_cellIds = thrust::device_ptr<uint>(dev_cellIds);
    dev_thrust_particleIds = thrust::device_ptr<uint>(dev_particleIds);

    cudaDeviceSynchronize();
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
    glBufferData(GL_ARRAY_BUFFER, num_fluid_particles*3*sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    //glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    //glEnableVertexAttribArray(0);
    //register VBO With Cuda
    registerVBO_WithCUDA(glVBO, &cudaVBO_resource);

    unregisterVBO_WithCUDA(cudaVBO_resource);
    glBindBuffer(GL_ARRAY_BUFFER, glVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, num_fluid_particles*3*sizeof(float), x);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    registerVBO_WithCUDA(glVBO, &cudaVBO_resource);

}

void FluidSimulator::RandomPositionStart()
{
    for (int i = 0; i < num_fluid_particles; i++)
    {
        float xCoord = randomFloatRangef(0.01f, 1.0f);
        float yCoord = randomFloatRangef(0.01f, 1.00f);
        float zCoord = randomFloatRangef(0.01f, 1.00f);

        x[i*3] = xCoord; x[i*3+1] = yCoord; x[i*3+2] = zCoord;
        //v[i*3] = 0.0f;   v[i*3+1] = 0.0f;   v[i*3+2] = 0.0f; 
        v[i] = glm::vec3(0.0f);
    }
}

__device__ int getCellNeighbors_2d(const int cell, const int gridWidth, const int gridWidth2, const int invCellGW, int *oCellNeighbors)
{
    int num_neighbors_found = 0;
    
    int factor = invCellGW;
    int gridUpperBound = gridWidth2 + factor*(gridWidth2);
    int gridLowerBound = factor*(gridWidth2);

    if (((cell - 1) % gridWidth) < (cell % gridWidth) && (cell-1)>=0)
    {
        
        if (cell-1-gridWidth >= gridLowerBound)
        {
            oCellNeighbors[num_neighbors_found] = cell-1-gridWidth;
            num_neighbors_found++;
        }
    }

    if (cell-gridWidth >= gridLowerBound)
    {
        oCellNeighbors[num_neighbors_found] = cell-gridWidth;
        num_neighbors_found++;
    }

    if (((cell + 1) % gridWidth) > (cell % gridWidth) && (cell+1) < gridUpperBound)
    {
        if (cell+1-gridWidth >= gridLowerBound)
        {
            oCellNeighbors[num_neighbors_found] = cell+1-gridWidth;
            num_neighbors_found++;
        }
    }

    if (((cell - 1) % gridWidth) < (cell % gridWidth) && (cell-1)>=0)
    {
        oCellNeighbors[num_neighbors_found] = cell-1;
        num_neighbors_found++;

    }
    oCellNeighbors[num_neighbors_found] = cell;

    num_neighbors_found++;
    if (((cell + 1) % gridWidth) > (cell % gridWidth) && (cell+1) < gridUpperBound)
    {
        oCellNeighbors[num_neighbors_found] = cell+1;
        num_neighbors_found++;

    }

    if (((cell - 1) % gridWidth) < (cell % gridWidth) && (cell-1)>=0)
    {
        if (cell-1+gridWidth < gridUpperBound)
        {
            oCellNeighbors[num_neighbors_found] = cell-1+gridWidth;
            num_neighbors_found++;
        }
    }

    if (cell+gridWidth < gridUpperBound)
    {  
        oCellNeighbors[num_neighbors_found] = cell+gridWidth;
        num_neighbors_found++;
    } 

    if (((cell + 1) % gridWidth) > (cell % gridWidth) && (cell+1) < gridUpperBound)
    {
        // oCellNeighbors[num_neighbors_found] = cell+1;
        // num_neighbors_found++;
        if (cell+1+gridWidth < gridUpperBound)
        {
            oCellNeighbors[num_neighbors_found] = cell+1+gridWidth;
            num_neighbors_found++;
        }

    }

    return num_neighbors_found;
}

__device__ int getCellNeighbors_3d(const int cell, const int gridWidth, const int gridWidth2, const int gridSize, int *oCellNeighbors)
{
    //printf("looking for the 3d neighbors of cell %d\n", cell);
    //int gridWidth2 = _gridWidth*gridWidth;
    int invCellGW = cell/gridWidth2;

    int neighbors1[9] = {0};
    int neighbors2[9] = {0};
    int neighbors3[9] = {0};

    for (int i = 0; i < 9; i++)
    {
        neighbors1[i] = 0;
        neighbors2[i] = 0;
        neighbors3[i] = 0;
    }

    
    int num_neighbors2 = 0;
    int num_neighbors3 = 0;

    if ((cell - gridWidth*gridWidth) >= 0)
    {
        num_neighbors3 = getCellNeighbors_2d(cell-gridWidth2, gridWidth, gridWidth2, invCellGW, neighbors3);
        for (int i = 0; i < 9; i++)
        {
            oCellNeighbors[i] = neighbors3[i];
        }
    }

    int num_neighbors1 = getCellNeighbors_2d(cell, gridWidth, gridWidth2, invCellGW, neighbors1);
    for (int i = 0; i < num_neighbors1; i++)
    {
        oCellNeighbors[num_neighbors3+i] = neighbors1[i];
    }

    if ((cell + gridWidth*gridWidth) < gridSize)
    {
        num_neighbors2 = getCellNeighbors_2d(cell+gridWidth2, gridWidth, gridWidth2, invCellGW, neighbors2);
        for (int i = 0; i < 9; i++)
        {
            oCellNeighbors[num_neighbors1+num_neighbors3+i] = neighbors2[i];
        }
    }

    return num_neighbors1 + num_neighbors2 + num_neighbors3;

}


__global__ void computeSpatialHash(const int n, const float inv_cell_size, const int gridWidth, const glm::vec3* ip, uint *oCellIds, uint *oParticleIds)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }
    //hash position
    int cellId = (int)(ip[index].x*inv_cell_size) + ((int)(ip[index].y*inv_cell_size))*gridWidth + ((int)(ip[index].z*inv_cell_size))*gridWidth*gridWidth;
    
    if (cellId >= 0)
    {
        oCellIds[index] = cellId;
        oParticleIds[index] = index;
    }
    
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

__global__ void explictEuler(int n, const float dt, float* ix, glm::vec3* ov, glm::vec3* op, glm::vec3* lastFrameHolder)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }   
    //store last frame for the update function at the end.
    lastFrameHolder[index] = glm::vec3(ix[index*3], ix[index*3+1], ix[index*3+2]);

    float velo_damp = 0.99f;
    float g = -9.8f;
    // if (op[index*4+1] <= 0)
    // {
    //     g = 0.0f;
    // } 
    float f_ext_x = 0.0f;
    float f_ext_y = g;
    float f_ext_z = 0.0f;

    //int i4 = index*4;

    ov[index] += dt*glm::vec3(0.0f, g, 0.0f);
    ov[index] *= velo_damp;
    // ov[index*3]   = ov[index*3]   + dt*f_ext_x;
    // ov[index*3+1] = ov[index*3+1] + dt*f_ext_y;
    // ov[index*3+2] = ov[index*3+2] + dt*f_ext_z;

    // ov[index*3] *= velo_damp;
    // ov[index*3+1] *= velo_damp;
    // ov[index*3+2] *= velo_damp;

    op[index] = glm::vec3(ix[index*3], ix[index*3+1], ix[index*3+2]) + dt*ov[index];

    // op[index*3]   = ix[index*3]   + dt*ov[index*3];
    // op[index*3+1] = ix[index*3+1] + dt*ov[index*3+1];
    // op[index*3+2] = ix[index*3+2] + dt*ov[index*3+2];

}

__global__ void computeDensity(int n, float h, int gridWidth, int gridWidth2, int gridSize, const glm::vec3* ip, const uint* cellIds, const uint* cellStarts, const uint* cellEnds, float* olambda)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

    float h2 = h*h;
    float h8 = h2*h2*h2*h2;
    float h6 = h2*h2*h2;


    float _pi = 3.141592f;
    //int particleId = particleIds[index];
    int cell = cellIds[index];

    float coeff = (4.0f)/(_pi*h8);
    float invRho0 = 1.0f/6378.0f;

    float L_coeff = (45.0f/((float)_pi*h6))*invRho0;

    //float coeff = 1.27324e8;

    glm::vec3 _ip = ip[index];

    // float ipx = ip[particleId*3];
    // float ipy = ip[particleId*3+1];
    // float ipz = ip[particleId*3+2];

    int neighboringCells[27] = {0};

    int num_neighbors_found = getCellNeighbors_3d(cell,  gridWidth, gridWidth*gridWidth, gridSize, neighboringCells);
    
    float rho = 0.0f; float sum_grad_C_i = 0.0f; float C_j_sum = 0.0f;
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
            //int particleId_i = particleIds[i];
            if (cellId_i != nCell)
            {
                //we've reached the end of the cell
                break;
            }

            // float rx = ipx - ip[particleId_i*3];
            // float ry = ipy - ip[particleId_i*3+1];
            // float rz = ipz - ip[particleId_i*3+2];

            glm::vec3 r = _ip - ip[i];

            float rd2 = r.x*r.x + r.y*r.y + r.z*r.z;
            if (rd2 < h2)
            {
                float W = coeff*(h2 - rd2)*(h2 - rd2)*(h2 - rd2);
                rho += W;

                float rd = sqrt(rd2);
                float dist2 = (h-rd)*(h-rd);
                float gradW_x = -L_coeff*(dist2*r.x);
                float gradW_y = -L_coeff*(dist2*r.y);
                float gradW_z = -L_coeff*(dist2*r.z);

                sum_grad_C_i += gradW_x*gradW_x + gradW_y*gradW_y + gradW_z*gradW_z;

            }
        }
    }

    float C_i = (rho*invRho0) - 1.0f;
    olambda[index] = -C_i/(sum_grad_C_i+600);

}

__global__ void projectDensityConstraint(int n, float h, int gridWidth,  int gridWidth2, int gridSize, float invRho0, glm::vec3* ip, glm::vec3* op, const uint* cellIds, const uint* cellStarts, const uint* cellEnds, const float* ilambda)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

    const float h2 = h*h;
    const float h6 = h2*h2*h2;

    const float _pi = 3.141592f;
    //const int particleId = particleIds[index];

    const float wCoeff = (15.0f/(_pi*h6));
    const float coeff = (45.0f/((float)_pi*h6));
    
    //float W_dq = wCoeff*powf((0.2f),3.0f);
    const float W_dq = wCoeff*0.2f*0.2f*0.2f;
    //const float s_corr = 0.0001f;

    const int cell = cellIds[index];

    const float lambda = ilambda[index];
    const glm::vec3 p = ip[index];

    int neighboringCells[27] = {0};

    const int num_neighbors_found = getCellNeighbors_3d(cell, gridWidth, gridWidth*gridWidth, gridSize, neighboringCells);
    //int num_neighbors_found = 1;

    float constraint_sum_x = 0.0f; float constraint_sum_y = 0.0f; float constraint_sum_z = 0.0f;  
    for (int k = 0; k < num_neighbors_found; k++)
    {
        const uint nCell = neighboringCells[k];
        const uint start = cellStarts[nCell];
        const uint end = cellEnds[nCell];

        if (start == end)
        {
            continue;
        }

        //loop through neighbors
            
        for (int i = start; i < end+1; i++)
        {
            const int cellId_i = cellIds[i];
            //const int particleId_i = particleIds[i];
            if (cellId_i != nCell)
            {
                //we've reached the end of the cell
                break;
            }

            const glm::vec3 r = p - ip[i];

            const float rd2 = r.x*r.x + r.y*r.y + r.z*r.z;
            if (rd2 < h2)
            {
                const float rd = sqrt(rd2);
                //float W_s = wCoeff*powf((h-rd),3.0f);
                const float W_s = wCoeff*(h-rd)*(h-rd)*(h-rd);
                const float s_corr = 0.1f*h*(W_s/W_dq)*(W_s/W_dq)*(W_s/W_dq)*(W_s/W_dq);

                const float dist2 = (h-rd)*(h-rd);
            
                const float gradW_x = -coeff*(dist2*r.x);
                const float gradW_y = -coeff*(dist2*r.y);
                const float gradW_z = -coeff*(dist2*r.z);

                const float lambda_sum = lambda + ilambda[i] + s_corr;
                
                constraint_sum_x += lambda_sum * gradW_x;
                constraint_sum_y += lambda_sum * gradW_y;
                constraint_sum_z += lambda_sum * gradW_z;
            }
        }
    }

    op[index] += glm::vec3(constraint_sum_x, constraint_sum_y, constraint_sum_z)*invRho0;

}

__global__ void updatePositions(int n, float dt, const glm::vec3 *lastFrame_x, const glm::vec3 *ip, glm::vec3 *ov, float *ox)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

     glm::vec3 newPos = ip[index];

     //ov[index] = (ip[index] - glm::vec3(ox[index*3], ox[index*3+1], ox[index*3+2]))/dt;
     ov[index] = (ip[index] - lastFrame_x[index])/dt;
    // //ox[index] = ip[index];

    // // ov[index*3] = (ip[index*3]-ox[index*3])/dt;
    // // ov[index*3+1] = (ip[index*3+1]-ox[index*3+1])/dt;
    // // ov[index*3+2] = (ip[index*3+2]-ox[index*3+2])/dt;
    
    // // ox[index*3] = ip[index*3];
    // // ox[index*3+1] = ip[index*3+1];
    // // ox[index*3+2] = ip[index*3+2];

    const float collDamp = 0.3f;
    float wall = 1.0f;
    float xWall = 1.0f;
    
    if (newPos.y < 0.0f && ov[index].y != 0.0f)
    {
        float tColl = (newPos.y-0.0f) / ov[index].y;

        newPos -= ov[index]*(1-collDamp)*tColl;

        newPos.y = 2.0f*0.0f-newPos.y;

        ov[index].y *= -1.0f;

        ov[index] *= collDamp;

    }
    // if (newPos.y > wall && ov[index].y != 0.0f)
    // {
    //     float tColl = (newPos.y-wall) / ov[index].y;

    //     newPos -= ov[index]*(1-collDamp)*tColl;

    //     newPos.y = 2.0f*wall-newPos.y;

    //     ov[index].y *= -1.0f;

    //     ov[index] *= collDamp;
    // }

    // if (newPos.x < 0.0f && ov[index].x != 0.0f)
    // {
    //     float tColl = (newPos.x-0.0f) / ov[index].x;

    //     newPos -= ov[index]*(1-collDamp)*tColl;

    //     newPos.x = 2.0f*0.0f-newPos.x;

    //     ov[index].x *= -1.0f;

    //     ov[index] *= collDamp;
    // }

    // if (newPos.z < 0.0f && ov[index].z != 0.0f)
    // {
    //     float tColl = (newPos.z-0.0f) / ov[index].z;

    //     newPos -= ov[index]*(1-collDamp)*tColl;

    //     newPos.z = 2.0f*0.0f-newPos.z;

    //     ov[index].z *= -1.0f;

    //     ov[index] *= collDamp;
    // }

    // if (newPos.x > wall && ov[index].x != 0.0f)
    // {
    //     float tColl = (newPos.x-wall) / ov[index].x;

    //     newPos -= ov[index]*(1-collDamp)*tColl;

    //     newPos.x = 2.0f*wall-newPos.x;

    //     ov[index].x *= -1.0f;

    //     ov[index] *= collDamp;
    // }

   
    // if (newPos.z > wall && ov[index].z != 0.0f)
    // {
    //     float tColl = (newPos.z-wall) / ov[index].z;

    //     newPos -= ov[index]*(1-collDamp)*tColl;

    //     newPos.z = 2.0f*wall-newPos.z;

    //     ov[index].z *= -1.0f;

    //     ov[index] *= collDamp;
    // }

    ox[index*3] = ip[index].x;
    ox[index*3+1] = ip[index].y;
    ox[index*3+2] = ip[index].z;

}

__global__ void sortSpatialArrays(int n, const uint* sortedParticleIds, const glm::vec3* ip, const glm::vec3* iv, const glm::vec3* lf, glm::vec3* op, glm::vec3* ov, glm::vec3* olf)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

    op[index] = ip[sortedParticleIds[index]];
    ov[index] = iv[sortedParticleIds[index]];
    olf[index] = lf[sortedParticleIds[index]];

}


void FluidSimulator::stepSimulation(const float dt)
{
    float* dev_x = (float*) mapGL(&cudaVBO_resource);

    int numThreads = 256;
    dim3 threadsPerBlock(numThreads);
    dim3 blocksPerGrid((num_fluid_particles+numThreads-1)/numThreads);

    timer().startGpuTimer();
    //call explict euler per particle. Modifies dev_v and dev_p
    explictEuler<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, dt, dev_x, dev_v, dev_p, dev_p_lastFrame);
    checkCUDAError("explicit euler failed");
    timer().endGpuTimer();
    float time = timer().getGpuElapsedTimeForPreviousOperation();
    printf("time for explict Euler = %f\n", time);
    cudaDeviceSynchronize();

    // //compute spatial hashes per particle. Modifies dev_CellIds and dev_particleIds
    timer().startGpuTimer();
    computeSpatialHash<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, invCellSize, gridWidth, dev_p, dev_cellIds, dev_particleIds);
    checkCUDAError("computeSpatialHash failed");
    timer().endGpuTimer();
    float time1 = timer().getGpuElapsedTimeForPreviousOperation();
    printf("time for compute spatial = %f\n", time1);
    cudaDeviceSynchronize();

    // //sort by cellId
    //timer().startGpuTimer();
    //thrustRadixSort(num_fluid_particles, dev_cellIds, dev_particleIds);
    // uint* unsorted_ParticleIds = dev_particleIds;
    // glm::vec3* unsorted_v = dev_v;
    // glm::vec3* unsorted_p = dev_p;

    cudaDeviceSynchronize();
    thrust::sort_by_key(dev_thrust_cellIds, dev_thrust_cellIds + num_fluid_particles, dev_thrust_particleIds);

    //cudaDeviceSynchronize();

////__global__ void sortSpatialArrays(int n, const uint* unsortedParticleIds, const uint* sortedParticleIds, const glm::vec3* ip, const glm::vec3* iv, glm::vec3* op, glm::vec3* ov)

    sortSpatialArrays<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, dev_particleIds, dev_p, dev_v, dev_p_lastFrame, dev_sorted_p, dev_sorted_v, dev_p_lastFrame_sorted);
    checkCUDAError("sortSpatialArrays failed");
    cudaDeviceSynchronize();

    // checkCUDAError("thrust error");
    // timer().endGpuTimer();
    // float time2 = timer().getGpuElapsedTimeForPreviousOperation();
    // printf("time for thrust sort = %f\n", time2);
    // cudaDeviceSynchronize();

    timer().startGpuTimer();
    // //get starting index of each cellId in the cellIds and particleIds parallel arrays and store in dev_cellStarts.
    findCellsInArray<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, gridWidth, dev_cellIds, dev_cellStarts, dev_cellEnds);
    checkCUDAError("findCellsInArray failed");
    timer().endGpuTimer();
    float time3 = timer().getGpuElapsedTimeForPreviousOperation();
    printf("time for find cells in array = %f\n", time3);

    cudaDeviceSynchronize();
    
    int num_iterations = 0;
    while (num_iterations < maxIterations)
    {
        timer().startGpuTimer();
        computeDensity<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, h, gridWidth, gridWidth2, gridSize, dev_sorted_p, dev_cellIds, dev_cellStarts, dev_cellEnds, dev_lambda);
        checkCUDAError("computeDensity failed");
        timer().endGpuTimer();
        float time4 = timer().getGpuElapsedTimeForPreviousOperation();
        printf("it %d computeDensity = %f\n", num_iterations, time4);
        
        //check that this is doing what I think.
        dev_p2 = dev_sorted_p;

        cudaDeviceSynchronize();
    
        timer().startGpuTimer();
        projectDensityConstraint<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, h, gridWidth, gridWidth2, gridSize, invRho0, dev_p2, dev_sorted_p, dev_cellIds, dev_cellStarts, dev_cellEnds, dev_lambda);
        checkCUDAError("projectDensityConstraint failed");
        timer().endGpuTimer();
        float time6 = timer().getGpuElapsedTimeForPreviousOperation();
        printf("it %d projectDensityConstraint = %f\n", num_iterations, time6);

        cudaDeviceSynchronize();

        num_iterations++;
    }


    updatePositions<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, dt, dev_p_lastFrame_sorted, dev_sorted_p, dev_sorted_v, dev_x);
    checkCUDAError("updatePositions failed");
    
    cudaDeviceSynchronize();

    //glm::vec3 *tmpV = dev_v;
    dev_v = dev_sorted_v;
    //dev_sorted_v = tmpV;

    //dev_p = dev_sorted_p;
    
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
    cudaFree(dev_sorted_p);
    cudaFree(dev_sorted_v);
    cudaFree(dev_p_lastFrame);
    cudaFree(dev_p_lastFrame_sorted);
    cudaFree(dev_p2);
    //cudaFree(dev_x);
    cudaFree(dev_v);
    cudaFree(dev_cellIds);
    cudaFree(dev_particleIds);
    cudaFree(dev_cellStarts);
    cudaFree(dev_cellEnds);

    cudaFree(dev_lambda);
}
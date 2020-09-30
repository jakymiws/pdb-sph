#include <glew.h>
#include "FluidSimulator.h"

#include "common.h"

#include <glm/glm.hpp>

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

thrust::device_ptr<uint> dev_thrust_cellIds;
thrust::device_ptr<uint> dev_thrust_particleIds;

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
    printf("openGL initialized in fluid simulator with glVBO = %d\n", glVBO);
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

    cudaMalloc((void**)&dev_lambda, num_fluid_particles*sizeof(float));
    checkCUDAError("malloc dev_lamba failed");
    
    cudaMemcpy(dev_v, v, num_fluid_particles*sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAError("memcpy v-->dev_v failed");

    cudaMalloc((void**)&dev_cellIds, num_fluid_particles*sizeof(uint));
    checkCUDAError("malloc failed");
    cudaMalloc((void**)&dev_particleIds, num_fluid_particles*sizeof(uint));
    checkCUDAError("malloc failed");
    cudaMalloc((void**)&dev_cellStarts, gridSize*sizeof(uint));
    checkCUDAError("malloc failed");
    cudaMalloc((void**)&dev_cellEnds, gridSize*sizeof(uint));
    checkCUDAError("malloc failed");
    cudaMalloc((void**)&dev_cellBounds, gridSize*sizeof(uint2));
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
        float xCoord = randomFloatRangef(0.5f, 1.5f);
        float yCoord = randomFloatRangef(0.01f, 0.5f);
        float zCoord = randomFloatRangef(0.5f, 1.5f);

        x[i*3] = xCoord; x[i*3+1] = yCoord; x[i*3+2] = zCoord;
        //v[i*3] = 0.0f;   v[i*3+1] = 0.0f;   v[i*3+2] = 0.0f; 
        v[i] = glm::vec3(0.0f);
    }
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

//Populates the cellBounds array with the index of the positions in each grid cell for easier access
__global__ void findCellsInArray(const int n, const int gridWidth, const uint* iCellIds, uint* cellStarts, uint* cellEnds, uint2* cellBounds)
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
        cellBounds[iCellIds[index]].x = 0;
        
    } else {
        if (iCellIds[index] != iCellIds[index-1])
        {
            cellStarts[iCellIds[index]] = index; 
            cellBounds[iCellIds[index]].x = index;
            
        }
    }

    if (index+1 < n)
    {
        if(iCellIds[index] != iCellIds[index+1])
        {
            cellEnds[iCellIds[index]] = index;
            cellBounds[iCellIds[index]].y = index;

        }
    } else {
        cellEnds[iCellIds[index]] = index;
        cellBounds[iCellIds[index]].y = index;
    }
}

__global__ void explictEuler(int n, const float dt, float* ix, glm::vec3* ov, glm::vec3* op, glm::vec3* lastFrameHolder)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }   
    //store last frame for the update function at the end of the simulation step.
    lastFrameHolder[index] = glm::vec3(ix[index*3], ix[index*3+1], ix[index*3+2]);

    float velo_damp = 0.99f;
    float g = -9.8f;

    float f_ext_x = 0.0f;
    float f_ext_y = g;
    float f_ext_z = 0.0f;

    ov[index] += dt*glm::vec3(0.0f, g, 0.0f);
    ov[index] *= velo_damp;

    op[index] = glm::vec3(ix[index*3], ix[index*3+1], ix[index*3+2]) + dt*ov[index];

}
__device__ glm::ivec3 get3DGridId(const float invCellSize, const glm::vec3 p)
{
    return glm::ivec3((int)(p.x*invCellSize), (int)(p.y*invCellSize), (int)(p.z*invCellSize));
}

__global__ void computeDensity_opt(int n, float h, int gridWidth, int gridWidth2, int gridSize, const glm::vec3* ip, const uint2* cellBounds, float* olambda)
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

    float coeff = (4.0f)/(_pi*h8);
    float invRho0 = 1.0f/6378.0f;

    float L_coeff = (45.0f/((float)_pi*h6))*invRho0;

    glm::vec3 _ip = ip[index];

    float invCellSize = 5.0f;

    glm::ivec3 gp = get3DGridId(invCellSize, _ip);

    float rho = 0.0f; float sum_grad_C_i = 0.0f; float C_j_sum = 0.0f;
    for (int _x = imax(0,gp.x-1); _x <= imin(gridSize-1,gp.x+1); _x++)
    {
        for (int _y = imax(0,gp.y-1); _y <= imin(gridSize-1,gp.y+1); _y++)
        {
            for (int _z = imax(0,gp.z-1); _z <= imin(gridSize-1,gp.z+1); _z++)
            {
                int neighboringCellId = _x + _y*gridWidth + _z*gridWidth*gridWidth;
                uint2 bounds = cellBounds[neighboringCellId];
                int start = bounds.x;
                int end = bounds.y;

                for (int i = start; i <= end; i++)
                {
                    glm::vec3 r = _ip - ip[i];
            
                    float rd2 = r.x*r.x + r.y*r.y + r.z*r.z;
                    if (rd2 < h2)
                    {
                        float W = coeff*(h2 - rd2)*(h2 - rd2)*(h2 - rd2);
                        rho += W;
            
                        float rd = sqrt(rd2);
                        float dist2 = (h-rd)*(h-rd);
            
                        r*= -L_coeff*dist2;
            
                        sum_grad_C_i += r.x*r.x + r.y*r.y + r.z*r.z;         
                    }                    
                }
            }
        }
    }

     float C_i = (rho*invRho0) - 1.0f;
     olambda[index] = -C_i/(sum_grad_C_i+600);

}

__global__ void projectDensityConstraint_opt(int n, float h, int gridWidth,  int gridWidth2, int gridSize, float invRho0, const glm::vec3* ip, glm::vec3* op, const uint2* cellBounds, const float* ilambda)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n)
    {
        return;
    }

    float h2 = h*h;
 
    float coeff = (45.0f/((float)3.141592f*h2*h2*h2));
    float s_corr = 0.0001f;

    float lambda = ilambda[index];
    glm::vec3 _ip = ip[index];

    float invCellSize = 5.0f;

    glm::ivec3 gp = get3DGridId(invCellSize, _ip);

    float constraint_sum_x = 0.0f; float constraint_sum_y = 0.0f; float constraint_sum_z = 0.0f;  

    for (int _x = imax(0,gp.x-1); _x <= imin(gridSize-1,gp.x+1); _x++)
    {
        for (int _y = imax(0,gp.y-1); _y <= imin(gridSize-1,gp.y+1); _y++)
        {
            for (int _z = imax(0,gp.z-1); _z <= imin(gridSize-1,gp.z+1); _z++)
            {
                int neighboringCellId = _x + _y*gridWidth + _z*gridWidth*gridWidth;
                uint2 bounds = cellBounds[neighboringCellId];
                uint start = bounds.x;
                uint end = bounds.y;

                for (uint i = start; i <= end; i++)
                {

                        glm::vec3 r = _ip - ip[i];
            
                        float rd2 = r.x*r.x + r.y*r.y + r.z*r.z;
                        if (rd2 < h2)
                        {
                            float rd = sqrt(rd2);

                            r *= -coeff*(h-rd)*(h-rd);

                            float lambda_sum = lambda + ilambda[i] + s_corr;
                
                            constraint_sum_x += lambda_sum * r.x;
                            constraint_sum_y += lambda_sum * r.y;
                        constraint_sum_z += lambda_sum * r.z;
                    }
                }
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
    float wall = 2.0f;
    float xWall = 2.0f;
    
    if (newPos.y < 0.0f && ov[index].y != 0.0f)
    {
        float tColl = (newPos.y-0.0f) / ov[index].y;

        newPos -= ov[index]*(1-collDamp)*tColl;

        newPos.y = 2.0f*0.0f-newPos.y;

        ov[index].y *= -1.0f;

        ov[index] *= collDamp;

    }
    if (newPos.y > wall && ov[index].y != 0.0f)
    {
        float tColl = (newPos.y-wall) / ov[index].y;

        newPos -= ov[index]*(1-collDamp)*tColl;

        newPos.y = 2.0f*wall-newPos.y;

        ov[index].y *= -1.0f;

        ov[index] *= collDamp;
    }

    if (newPos.x < 0.0f && ov[index].x != 0.0f)
    {
        float tColl = (newPos.x-0.0f) / ov[index].x;

        newPos -= ov[index]*(1-collDamp)*tColl;

        newPos.x = 2.0f*0.0f-newPos.x;

        ov[index].x *= -1.0f;

        ov[index] *= collDamp;
    }

    if (newPos.z < 0.0f && ov[index].z != 0.0f)
    {
        float tColl = (newPos.z-0.0f) / ov[index].z;

        newPos -= ov[index]*(1-collDamp)*tColl;

        newPos.z = 2.0f*0.0f-newPos.z;

        ov[index].z *= -1.0f;

        ov[index] *= collDamp;
    }

    if (newPos.x > wall && ov[index].x != 0.0f)
    {
        float tColl = (newPos.x-wall) / ov[index].x;

        newPos -= ov[index]*(1-collDamp)*tColl;

        newPos.x = 2.0f*wall-newPos.x;

        ov[index].x *= -1.0f;

        ov[index] *= collDamp;
    }

   
    if (newPos.z > wall && ov[index].z != 0.0f)
    {
        float tColl = (newPos.z-wall) / ov[index].z;

        newPos -= ov[index]*(1-collDamp)*tColl;

        newPos.z = 2.0f*wall-newPos.z;

        ov[index].z *= -1.0f;

        ov[index] *= collDamp;
    }

    ox[index*3] = newPos.x;
    ox[index*3+1] = newPos.y;
    ox[index*3+2] = newPos.z;

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

    //call explict euler per particle. Modifies dev_v and dev_p
    explictEuler<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, dt, dev_x, dev_v, dev_p, dev_p_lastFrame);
    checkCUDAError("explicit euler failed");

    // //compute spatial hashes per particle. Modifies dev_CellIds and dev_particleIds
    computeSpatialHash<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, invCellSize, gridWidth, dev_p, dev_cellIds, dev_particleIds);
    checkCUDAError("computeSpatialHash failed");
  
    thrust::sort_by_key(dev_thrust_cellIds, dev_thrust_cellIds + num_fluid_particles, dev_thrust_particleIds);
    
    sortSpatialArrays<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, dev_particleIds, dev_p, dev_v, dev_p_lastFrame, dev_sorted_p, dev_sorted_v, dev_p_lastFrame_sorted);
    checkCUDAError("sortSpatialArrays failed");

    // //get starting index of each cellId in the cellIds and particleIds parallel arrays and store in dev_cellStarts.
    findCellsInArray<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, gridWidth, dev_cellIds, dev_cellStarts, dev_cellEnds, dev_cellBounds);
    checkCUDAError("findCellsInArray failed");

    int num_iterations = 0;
    while (num_iterations < maxIterations)
    {
        computeDensity_opt<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, h, gridWidth, gridWidth2, gridSize, dev_sorted_p, dev_cellBounds, dev_lambda);
        checkCUDAError("computeDensity failed");
        
        dev_p2 = dev_sorted_p;
    
        projectDensityConstraint_opt<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, h, gridWidth, gridWidth2, gridSize, invRho0, dev_p2, dev_sorted_p, dev_cellBounds, dev_lambda);
        checkCUDAError("projectDensityConstraint failed");

        cudaDeviceSynchronize();

        num_iterations++;
    }


    updatePositions<<<blocksPerGrid, threadsPerBlock>>>(num_fluid_particles, dt, dev_p_lastFrame_sorted, dev_sorted_p, dev_sorted_v, dev_x);
    checkCUDAError("updatePositions failed");
    
    dev_v = dev_sorted_v;

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
    cudaFree(dev_cellBounds);

    cudaFree(dev_lambda);
}
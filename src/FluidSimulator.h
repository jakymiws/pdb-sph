#include <iostream>
#include <vector>
#include <unordered_map>
#include <list>

#include<cuda.h>
#include<cuda_runtime.h>
#include <cuda_gl_interop.h>

class FluidSimulator
{
private:
    void RandomPositionStart();

    //void hashParticlePositions();
    //int spatial_hash(const float x, const float y, const float z);

    float randomFloatf();
    float randomFloatRangef(float a, float b);

    int num_fluid_particles;

    float* x;
    float* p;
    float* v;

    float *dev_v;
    float *dev_p;
    //float *dev_x;

    uint *dev_cellIds;
    uint *dev_particleIds;
    uint *dev_cellStarts;

    //hashing
    int cellSize;
    float invCellSize;
    int gridWidth;
    int gridSize;

    //rendering
    void InitGL();
    void AllocCudaArrays();

    unsigned int glVBO, glVAO;
    struct cudaGraphicsResource *cudaVBO_resource;


public:
    FluidSimulator(int n, int _cellSize, int _gridWidth);
    ~FluidSimulator(){};

    void stepSimulation(const float dt);
    void cleanUpSimulation();
    
    //rendering
    uint getVBO();
    int getNumFluidParticles();
    //I feel like it should be possible to return a VAO instead if I bind it correctly in the
    //constructor. main.cpp shouldn't care directly right?



};



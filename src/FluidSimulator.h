#include <iostream>
#include <vector>
#include <unordered_map>
#include <list>

#include<cuda.h>
#include<cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>


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
    glm::vec3* p;
    glm::vec3* v;

    glm::vec3 *dev_v;
    glm::vec3 *dev_p;
    glm::vec3 *dev_sorted_p;
    glm::vec3 *dev_sorted_v;
    glm::vec3 *dev_p_lastFrame_sorted;
    glm::vec3 *dev_p2;
    glm::vec3 *dev_p_lastFrame;
    //float *dev_x;
    //float *dev_density;
    float *dev_lambda;

    uint *dev_cellIds;
    uint *dev_particleIds;
    uint *dev_cellStarts;
    uint *dev_cellEnds;

    float h;
    float rho0;
    float invRho0;
    float epsR;

    int maxIterations;

    //hashing
    float cellSize;
    float invCellSize;
    int gridWidth;
    int gridWidth2;
    int gridSize;

    //rendering
    void InitGL();
    void AllocCudaArrays();

    unsigned int glVBO, glVAO;
    struct cudaGraphicsResource *cudaVBO_resource;


public:
    FluidSimulator(int n, float _cellSize, int _gridWidth);
    ~FluidSimulator(){};

    void stepSimulation(const float dt);
    void cleanUpSimulation();
    
    //rendering
    uint getVBO();
    int getNumFluidParticles();
    //I feel like it should be possible to return a VAO instead if I bind it correctly in the
    //constructor. main.cpp shouldn't care directly right?



};



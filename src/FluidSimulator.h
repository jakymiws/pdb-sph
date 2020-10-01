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
    float randomFloatf();
    float randomFloatRangef(float a, float b);
    
    void cleanUpSimulation();

    int num_fluid_particles;

    //host spatial vars
    float* x;
    glm::vec3* p;
    glm::vec3* v;

    //device spatial vars
    glm::vec3 *dev_v;
    glm::vec3 *dev_p;
    glm::vec3 *dev_sorted_p;
    glm::vec3 *dev_sorted_v;
    glm::vec3 *dev_p_lastFrame_sorted;
    glm::vec3 *dev_p2;
    glm::vec3 *dev_p_lastFrame;

    float *dev_lambda;

    uint *dev_cellIds;
    uint *dev_particleIds;
   // uint *dev_cellStarts;
   // uint *dev_cellEnds;
    uint2 *dev_cellBounds;

    //simulation constants
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
    ~FluidSimulator(){ cleanUpSimulation(); };

    void stepSimulation(const float dt);
    
    //rendering
    uint getVBO();
    int getNumFluidParticles();

};



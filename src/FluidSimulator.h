#include <iostream>
#include <vector>
#include <unordered_map>
#include <list>

#include<cuda.h>
#include<cuda_runtime.h>

class FluidSimulator
{
private:
    void RandomPositionStart();

    void hashParticlePositions();
    int spatial_hash(const float x, const float y, const float z);

    float randomFloatf();
    float randomFloatRangef(float a, float b);

    int num_fluid_particles;

    std::unordered_map<int, std::list<int>> spatialHashMap;
    float4* x;
    float4* p;
    float4* v;

    //hashing
    int cell_size = 1.0f;
    float inv_cell_size = 1.0/1.0f;
    int gridWidth = 4;

public:
    FluidSimulator(int n);
    ~FluidSimulator(){};

    void predictSimStepGPU();



};



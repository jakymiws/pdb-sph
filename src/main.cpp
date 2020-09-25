#include <glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <unordered_map>
#include <list>

#include "LoadShaders.h"
#include "camera.h"

//TODO: delete reference to this in makefile
#include "FluidSimulator.h"

//TODO: 
//1.Move sim to cuda
//2. Move neighbor finding to cuda
    //2a. Will need to figure out how to get cuda and opengl to work together - there was a function for it I saw on cuda particles I think
//3. Test 3D

struct IndirectRenderParams
{
    GLuint count;
    GLuint primCount;
    GLuint firstIndex;
    GLuint baseVertex;
    GLuint baseInstance;
};


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
void compute_density();

const int SCREEN_WIDTH = 1280;
const int SCREEN_HEIGHT = 720;

// timing
bool debugMode = false;
float deltaTime = 0.0f;	// time between current frame and last frame
float lastTime = 0.0f;
int numFrames = 0;
 
// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCREEN_WIDTH / 2.0f;
float lastY = SCREEN_HEIGHT / 2.0f;
bool firstMouse = true;

glm::vec3 lightPos(0.0f, 1.0f, 3.0f);

float zTrans = 0.0f;
float xTrans = 0.0f;
float qScale = 10.0f;

int debugSwitch = 0;

//simulation settings and variables
int num_fluid_particles = 600;
int maxIterations = 4;

const float gravity_accel = -9.8f;
const float velo_damp = 0.99f;
float dt = 0.0083f;

float mass = 1.0f;
float wMass = 1.0f/mass;

glm::vec3* p;
glm::vec3* x;
glm::vec3* v;

float h = 0.1f;
float density0 = 6378.0f;
float* density;
float* lambda;

float bboxDim = 4.0f;
float epsR = 600.0f;

int cell_size = 1;
float inv_cell_size = 1.0f/1.0f;

std::unordered_map<int, std::list<int>> gridHashMap;

int gridWidth = 4;

float randomFloat()
{
    return (rand() / (float)RAND_MAX);
}

float randomFloatRange(float a, float b)
{
    float f = randomFloat();
    float diff = b-a;
    float r = f*diff;
    return a+r;
}

int spatial_hash(glm::vec3 pos)
{
    return (int)pos.x*inv_cell_size + ((int)pos.y*inv_cell_size)*gridWidth;
}

void hashParticlePositions()
{
    for (int i = 0; i < num_fluid_particles; i++)
    {
        //I think it should be p[i] and not x[i] since I'm only calling this at the moment after the predict step.
        int gridCell = spatial_hash(p[i]);
        gridHashMap[gridCell].push_front(i);
    }
}

void init_fluid()
{
    x = new glm::vec3[num_fluid_particles];
    p = new glm::vec3[num_fluid_particles];
    v = new glm::vec3[num_fluid_particles];

    for (int i = 0; i < num_fluid_particles; i++)
    {
        // float xCoord = fabs((randomFloat() - 0.5f) * bboxDim);
        // float yCoord = fabs((randomFloat() - 0.5f) * bboxDim);
        // float zCoord = fabs((randomFloat() - 0.5f) * bboxDim);
        float xCoord = randomFloatRange(0.0f, 0.5f);
        float yCoord = randomFloatRange(0.0f, 0.5f);
        float zCoord = randomFloatRange(0.0f, 0.5f);
        
        x[i] = glm::vec3(xCoord, yCoord, 0.0f);
        v[i] = glm::vec3(0.0f);
    }

    lambda = new float[num_fluid_particles];
    density = new float[num_fluid_particles];
}

std::vector<int> findNeighbors(int pidx)
{
    int index = spatial_hash(p[pidx]);
  
    //get neighboring gridCells
    std::vector<int> neighboringCells;
    neighboringCells.push_back(index);  
    if (((index + 1) % gridWidth) > (index % gridWidth) && (index+1) < gridWidth*gridWidth)
    {
        neighboringCells.push_back(index+1);
        neighboringCells.push_back(index+1+gridWidth); neighboringCells.push_back(index+1-gridWidth);
    }
    
    if (((index - 1) % gridWidth) < (index % gridWidth) && (index-1)>=0)
    {
        neighboringCells.push_back(index-1); 
        neighboringCells.push_back(index-1+gridWidth); neighboringCells.push_back(index-1-gridWidth);
    }
    
    neighboringCells.push_back(index+gridWidth);
    neighboringCells.push_back(index-gridWidth); 

    std::vector<int> neighboringParticles;
    //then append their contents to the neighboring particles vector.
    for (int i = 0; i < neighboringCells.size(); i++)
    {   
        int ni = neighboringCells[i]; 

        if (ni >= 0 && ni <= (gridWidth*gridWidth)-1)
        {
            std::list<int> l = gridHashMap[ni];
            for (int const& j : l)
            {
                neighboringParticles.push_back(j);
            }
        }
    }
    
    return neighboringParticles;
}

void predict_sim_step()
{
    glm::vec3 f_ext = glm::vec3(0.0f, gravity_accel, 0.0f);

    gridHashMap.clear();

    //predict v using explicit euler and then damp velocities
    for (int i = 0; i < num_fluid_particles; i++)
    {
        v[i] = v[i] + dt*f_ext;
        v[i] *= velo_damp;  

        p[i] = x[i] + dt*v[i];
    }

    //predict positifindNeighbors(5).size()on using explicit euler
    // for (int i = 0; i < num_fluid_particles; i++)
    // {
    //     p[i] = x[i] + dt*v[i];
    // }

    //update hashtable with predictions
    hashParticlePositions();

    // for (int i = 0; i < num_fluid_particles; i++)
    // {
    //     int gc = spatial_hash(p[i]);
    //     std::list<int> ll = gridHashMap[gc];
    //     printf("gridCell %d has the following particles:\n", gc);
    //     for (int const& i : ll)
    //     {
    //         printf(" %d \n", i);
    //     }
    //     printf("---end cell %d---\n", gc);
    // }

    // for (int i = 0; i < num_fluid_particles; i++)
    // {
    //     std::vector<int> neighbors = findNeighbors(i);
    //     for (int j = 0; j < neighbors.size(); j++)
    //     {
    //         int ngc = spatial_hash(p[neighbors[j]]);
    //         printf("%d is in cell %d and has %d which is in cell %d\n", i, spatial_hash(p[i]) , neighbors[j], ngc);
    //     }

    // }
    
}

//TODO:
void genCollisionConstraints()
{
    //Generate the relevant collision constraints from the predicted positions
}

void computeDensity_naive()
{
    float h8 = powf(h,8.0f);
    float h2 = powf(h, 2.0f);
    //density
    for (int i = 0; i < num_fluid_particles; i++)
    {
        //NAIVE SEARCH - CHANGE AFTER GETTING BASICS WORKING
        float rhoi = 0.0f;
        //implement loop over these hashed neighbors
        //std::vector<int> neighbors = findNeighbors(i);
        for (int j = 0; j < num_fluid_particles; j++)
        {
            glm::vec3 r = p[i] - p[j];
            float rd = glm::length(r);

            if (rd*rd < h2)
            {
                float W = (4.0f)/(M_PI*h8)*powf((h2 - rd*rd),3.0f);
                rhoi += mass*W;
            }
        }
        density[i] = rhoi;
        //printf("density[%d] = %f\n", i, density[i]);
    }
    
    //constraint force (lambda in the paper)
    for (int i = 0; i < num_fluid_particles; i++)
    {
        float C_i = (density[i]/density0) - 1.0f;
        //C_i = glm::max(C_i, 0.0f);
        float sum_grad_C_i = 0.0f;
        for (int j = 0; j < num_fluid_particles; j++)
        {
            glm::vec3 r = p[i] - p[j];
            float rd = glm::length(r);
            if (rd*rd < h2)
            {
                glm::vec3 gradW = -(45.0f/((float)M_PI*h2*h2*h2))*((h-rd)*(h-rd)*r);
                //glm::vec3 gradW = -(30.0f/((float)M_PI*h2*h2))*(((1-0.5f)*(1-0.5f)*r)/0.5f);

                gradW /= density0;
                float gradWd = glm::length(gradW);
                sum_grad_C_i += gradWd * gradWd;
                //printf("gradWd[%d] = %f \n",j, gradWd);
            }
        }
       // printf("sum[%d] = %f\n", i, sum_grad_C_i);
        lambda[i] = -C_i/(sum_grad_C_i+epsR);
    }
}

void computeDensity()
{
    float h8 = powf(h,8.0f);
    float h2 = powf(h, 2.0f);
    //density
    for (int i = 0; i < num_fluid_particles; i++)
    {
        float rhoi = 0.0f;
        //implement loop over these hashed neighbors
        std::vector<int> neighbors = findNeighbors(i);
        for (int j = 0; j < neighbors.size(); j++)
        {
            glm::vec3 r = p[i] - p[neighbors[j]];
            float rd = glm::length(r);

            if (rd*rd < h2)
            {
                float W = (4.0f)/(M_PI*h8)*powf((h2 - rd*rd),3.0f);
                rhoi += mass*W;
            }
        }
        density[i] = rhoi;
        //printf("density[%d] = %f\n", i, density[i]);
    }
    
    //constraint force (lambda in the paper)
    for (int i = 0; i < num_fluid_particles; i++)
    {
        float C_i = (density[i]/density0) - 1.0f;
        //C_i = glm::max(C_i, 0.0f);
        float sum_grad_C_i = 0.0f;
        std::vector<int> neighbors = findNeighbors(i);
        for (int j = 0; j < neighbors.size(); j++)
        {
            glm::vec3 r = p[i] - p[neighbors[j]];
            float rd = glm::length(r);
            if (rd*rd < h2)
            {
                glm::vec3 gradW = -(45.0f/((float)M_PI*h2*h2*h2))*((h-rd)*(h-rd)*r);
                //glm::vec3 gradW = -(30.0f/((float)M_PI*h2*h2))*(((1-0.5f)*(1-0.5f)*r)/0.5f);

                gradW /= density0;
                float gradWd = glm::length(gradW);
                sum_grad_C_i += gradWd * gradWd;
                //printf("gradWd[%d] = %f \n",j, gradWd);
            }
        }
       // printf("sum[%d] = %f\n", i, sum_grad_C_i);
        lambda[i] = -C_i/(sum_grad_C_i+epsR);
    }
}

void projectDensityConstraint_naive()
{
    float h2 = h*h;
    for (int i = 0; i < num_fluid_particles; i++)
    {   
        glm::vec3 constraint_sum = glm::vec3(0.0f);
        for (int j = 0; j < num_fluid_particles; j++)
        {
            glm::vec3 r = p[i] - p[j];
            float rd = glm::length(r);

            if (sqrtf(rd*rd) < sqrtf(h2))
            {
                float W_s = (15.0f/(M_PI*h2*h2*h2))*powf((h-rd),3.0f);
                float W_dq = (15.0f/(M_PI*h2*h2*h2))*powf((0.2f),3.0f);
                float s_corr = 0.1*h*powf(W_s/W_dq, 4); 
                glm::vec3 gradW = -(45.0f/((float)M_PI*h2*h2*h2))*((h-rd)*(h-rd)*r);
                //printf("s_corr[%d, %d] = %f\n", i,j, s_corr);
               // glm::vec3 gradW = -(30.0f/((float)M_PI*h2*h2))*(((1-0.1f)*(1-0.1f)*r)/0.1f);
                constraint_sum += (lambda[i] + lambda[j] + s_corr) * gradW;
            }
        }
        
        glm::vec3 dPi = constraint_sum/density0;
        p[i] += dPi;
    }
}


void projectDensityConstraint()
{
    float h2 = h*h;
    for (int i = 0; i < num_fluid_particles; i++)
    {   
        glm::vec3 constraint_sum = glm::vec3(0.0f);
        std::vector<int> neighbors = findNeighbors(i);
        for (int j = 0; j < neighbors.size(); j++)
        {
            glm::vec3 r = p[i] - p[neighbors[j]];
            float rd = glm::length(r);

            if (sqrtf(rd*rd) < sqrtf(h2))
            {
                float W_s = (15.0f/(M_PI*h2*h2*h2))*powf((h-rd),3.0f);
                float W_dq = (15.0f/(M_PI*h2*h2*h2))*powf((0.2f),3.0f);
                float s_corr = 0.1*h*powf(W_s/W_dq, 4); 
                glm::vec3 gradW = -(45.0f/((float)M_PI*h2*h2*h2))*((h-rd)*(h-rd)*r);
                //printf("s_corr[%d, %d] = %f\n", i,j, s_corr);
               // glm::vec3 gradW = -(30.0f/((float)M_PI*h2*h2))*(((1-0.1f)*(1-0.1f)*r)/0.1f);
                constraint_sum += (lambda[i] + lambda[neighbors[j]] + s_corr) * gradW;
            }
        }
        
        glm::vec3 dPi = constraint_sum/density0;
        p[i] += dPi;
    }
}

void applyVorticity()
{

}

void applyViscosity()
{

}

void solve()
{
    int iteration = 0;
    while (iteration < maxIterations)
    {
        computeDensity();
        projectDensityConstraint();
        //detectCollisions();
        iteration++;
    }

    //apply the constrained positions
    for (int i = 0; i < num_fluid_particles; i++)
    {
        v[i] = (p[i]-x[i])/dt;
        if (p[i].y <= 0 )
        {
            v[i] *= -0.3f;
        }

        applyVorticity();
        applyViscosity();
        x[i] = p[i];
    }
}

int main(void)
{
    if (!glfwInit())
        exit(EXIT_FAILURE);

    init_fluid();

    GLFWwindow* window;
    GLuint vertex_buffer, vertex_shader, fragment_shader, program;
    GLint mvp_location, vpos_location, vcol_location;
  
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
 
    window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "PDB SPH", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
  
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    glfwSwapInterval(1);

    glewInit();

    FluidSimulator *fs = new FluidSimulator(1000, cell_size, gridWidth);
    //while true:
    // for (int i = 0; i < 3; i++)
    // {
    //     fs->stepSimulation(dt);
    //     printf("i=%d\n", i);
    //     uint cvbo = fs->getVBO();
    //     int cn = fs->getNumFluidParticles();
    //     printf("cvbo = %d\n", cvbo);
    //     printf("cn = %d\n", cn);
    // }
    
    //     //setVertexBuffer(fs->getBuffer(), fs->getNumParticles());
    //     //gl display code
    //     fs->cleanUpSimulation();
    // delete fs;
    // glfwDestroyWindow(window);
 
    //glfwTerminate();
    //exit(EXIT_SUCCESS);
    //return 0;

    glEnable(GL_DEPTH_TEST);

    ShaderInfo light_shader_info[] = {
        {GL_VERTEX_SHADER, "../shaders/light.vs"},
        {GL_FRAGMENT_SHADER, "../shaders/light.fs"},
        {GL_NONE, NULL}
    };

    GLuint light_shader_program = LoadShaders(light_shader_info);

    printf("light shader program: %d \n", light_shader_program);

float verts[] = {
    -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
     0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 
     0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 
     0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 
    -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 
    -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 

    -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
     0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
     0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
     0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
    -0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,

    -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
    -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
    -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
    -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
    -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
    -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,

     0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
     0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
     0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
     0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
     0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
     0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,

    -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
     0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
     0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
     0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,

    -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
     0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
     0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
     0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
    -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
    -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f
};

    // //light cube
    // unsigned int lightVAO, lightVBO;
    // glGenVertexArrays(1, &lightVAO);
    // glGenBuffers(1, &lightVBO);
    // printf("light vbo = %d\n", lightVBO);

    // glBindVertexArray(lightVAO);
    // glBindBuffer(GL_ARRAY_BUFFER, lightVBO);
    // glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0);
    // glEnableVertexAttribArray(0);

    //ground plane
    // float quad_vertices[] = {
    //     -0.5f, -0.5f,
    //     0.5f, -0.5f,
    //     0.5f, 0.5f,
    //     -0.5f, 0.5f
    // };

    // ShaderInfo ground_plane_shader_info[] = {
    //     {GL_VERTEX_SHADER, "../shaders/ground.vs"},
    //     {GL_FRAGMENT_SHADER, "../shaders/ground.fs"},
    //     {GL_NONE, NULL}
    // };

   // GLuint ground_shader_program = LoadShaders(ground_plane_shader_info);

  //  printf("ground shader program: %d \n", ground_shader_program);

    // unsigned int groundVAO, groundVBO;
    // glGenVertexArrays(1, &groundVAO);
    // glGenBuffers(1, &groundVBO);

    // glBindVertexArray(groundVAO);
    // glBindBuffer(GL_ARRAY_BUFFER, groundVBO);
    // glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);

    // glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
    // glEnableVertexAttribArray(0);
    while (!glfwWindowShouldClose(window))
    {
        float currentTime = glfwGetTime();
        deltaTime = currentTime - lastTime;
        numFrames++;
        if (deltaTime >= 1.0)
        {   
            if (debugMode)
            {
                printf("fps = %f\n", (float)numFrames);
                printf("mpf = %f\n", 1000.0f/(float)numFrames);
            }
            
            numFrames = 0;
            lastTime = currentTime;
        }
        
        processInput(window);

        float ratio;
        int width, height;

        glfwGetFramebufferSize(window, &width, &height);
        ratio = width / (float) height;

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        fs->stepSimulation(dt);
        //printf("i=%d\n", numFrames);
        uint cvbo = fs->getVBO();
        int cn = fs->getNumFluidParticles();
        //return 0;
        //printf("cvbo = %d\n", cvbo);
        //printf("cn = %d\n", cn);

        //float time1 = glfwGetTime();
        //predict_sim_step();
        //return 0;
        //float time2 = glfwGetTime();
        //printf("predict_sim_step() took %f to complete\n", time2-time1);
        //printf("(%f%f,%f)\n", x[4].x, x[4].y, x[4].z);
        //return 0;
        //genCollisionConstraints();
        //float time3 = glfwGetTime();
        //solve();
        //float time4 = glfwGetTime();
        //printf("solve() took %f to complete\n", time4-time3);

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 model = glm::mat4(1.0f);
        //Display light cube
        glUseProgram(light_shader_program);

        glUniformMatrix4fv(glGetUniformLocation(light_shader_program, "projection"), 1, GL_FALSE, &projection[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(light_shader_program, "view"), 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(light_shader_program, "model"), 1, GL_FALSE, &model[0][0]);

        glBindBuffer(GL_ARRAY_BUFFER, cvbo);
        glVertexPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);
        glDrawArrays(GL_POINTS, 0, cn);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDisableClientState(GL_VERTEX_ARRAY);
        
        // glBindVertexArray(lightVAO);
        
        // for (int i = 0; i < num_fluid_particles; i++)
        // {
        //     glm::mat4 model = glm::mat4(1.0f);
        //     model = glm::mat4(1.0f);
        //     //model = glm::translate(model, lightPos + glm::vec3(i,0,0));
        //     model = glm::translate(model, x[i]);
        //     model = glm::scale(model, glm::vec3(0.01f));
        //     if (i == 15)
        //     {
        //         //model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        //         glUniform1f(glGetUniformLocation(light_shader_program, "inCol"), 1.0f);

        //     } else {
        //         glUniform1f(glGetUniformLocation(light_shader_program, "inCol"), 0.5f);
        //     }

        //     glUniformMatrix4fv(glGetUniformLocation(light_shader_program, "model"), 1, GL_FALSE, &model[0][0]);

        //     glDrawArrays(GL_TRIANGLES, 0, 36);
        // }
       
        //End Display light cube

        // //Display ground plane
        // glUseProgram(ground_shader_program);

        // glUniformMatrix4fv(glGetUniformLocation(ground_shader_program, "projection"), 1, GL_FALSE, &projection[0][0]);
        // glUniformMatrix4fv(glGetUniformLocation(ground_shader_program, "view"), 1, GL_FALSE, &view[0][0]);
        
        // model = glm::mat4(1.0f);
        // //model = glm::translate(model, glm::vec3(xTrans, 0, zTrans));
        // model = glm::scale(model, glm::vec3(qScale));
        // model = glm::rotate(model, 1.5708f, glm::vec3(1,0,0)); //1.5708 radians = 90 degrees

        // glUniformMatrix4fv(glGetUniformLocation(ground_shader_program, "model"), 1, GL_FALSE, &model[0][0]);

        // glBindVertexArray(groundVAO);
        // glDrawArrays(GL_QUADS, 0, 4);
        //End Display ground plane

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
 
    delete[] x;
    delete[] v;
    delete[] p;
    delete[] lambda;
    delete[] density;

    fs->cleanUpSimulation();
    delete fs;

    glfwDestroyWindow(window);
 
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

void processInput(GLFWwindow *window)
{
    float offset = 0.01f;
    float dt = 0.01f;

     if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
    {
        debugSwitch = 1;
        printf("debugSwitch on: %d \n", debugSwitch);
    }
     if (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS)
    {
        debugSwitch = 0;
        printf("debugSwitch on: %d \n", debugSwitch);
    }

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, dt);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, dt);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, dt);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, dt);
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
    {
        xTrans += offset;
        printf("X: %f \n", xTrans);
    }
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
    {
        xTrans -= offset;
        printf("X: %f \n", xTrans);
    }
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
    {
        zTrans += offset;
        printf("Z: %f \n", zTrans);
    }
    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS)
    {
        zTrans -= offset;
        printf("Z: %f \n", zTrans);
    }

    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
    {
        debugMode = !debugMode;
        printf("-------------------\n");
    }

        
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}
#include <glew.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>

#include "LoadShaders.h"
#include "camera.h"

struct IndirectRenderParams
{
    GLuint count;
    GLuint primCount;
    GLuint firstIndex;
    GLuint baseVertex;
    GLuint baseInstance;
};

struct ClothEdge
{
    int e1;
    int e2;
    float k;
    float l0;
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

const float planeDim = 50.0f;

float zTrans = 0.0f;
float xTrans = 0.0f;
float qScale = 10.0f;

int debugSwitch = 0;

int num_fluid_particles = 1000;
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

float bboxDim = 1.0f;
float epsR = 600.0f;

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

void init_fluid()
{
    x = new glm::vec3[num_fluid_particles];
    p = new glm::vec3[num_fluid_particles];
    v = new glm::vec3[num_fluid_particles];

    for (int i = 0; i < num_fluid_particles; i++)
    {
        float xCoord = (randomFloat() - 0.5f) * bboxDim;
        float yCoord = (randomFloat() - 0.5f) * bboxDim;
        float zCoord = (randomFloat() - 0.5f) * bboxDim;
        
        x[i] = glm::vec3(xCoord, yCoord, 0.0f);
        v[i] = glm::vec3(0.0f);

    }

    lambda = new float[num_fluid_particles];
    density = new float[num_fluid_particles];
}

void predict_sim_step()
{
    glm::vec3 f_ext = glm::vec3(0.0f, gravity_accel, 0.0f);

    //predict v using explicit euler and damp velocities
    for (int i = 0; i < num_fluid_particles; i++)
    {
        v[i] = v[i] + dt*f_ext;
        v[i] *= velo_damp;   
    }

    //predict position using explicit euler
    for (int i = 0; i < num_fluid_particles; i++)
    {
        p[i] = x[i] + dt*v[i];
    }
}

//TODO:
void genCollisionConstraints()
{
    //Generate the relevant collision constraints from the predicted positions
}

//TODO:
void findNeighbors()
{

}

void computeDensity()
{
    float h8 = powf(h,8.0f);
    float h2 = powf(h, 2.0f);
    //density
    for (int i = 0; i < num_fluid_particles; i++)
    {
        //NAIVE SEARCH - CHANGE AFTER GETTING BASICS WORKING
        float rhoi = 0.0f;
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

void projectDensityConstraint()
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

void applyVorticity()
{

}

void applyViscosity()
{

}

void solve()
{
    //while (i < maxIterations)
        //for each constraint C_i
            //projectConstraint C_i
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
        if (p[i].y <= -bboxDim )
        {
            //printf("pvy %f\n",p[i].y);

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
 
    window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Real Time Grass Simulator", NULL, NULL);
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

    //light cube
    unsigned int lightVAO, lightVBO;
    glGenVertexArrays(1, &lightVAO);
    glGenBuffers(1, &lightVBO);

    glBindVertexArray(lightVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lightVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    //ground plane
    float quad_vertices[] = {
        -0.5f, -0.5f,
        0.5f, -0.5f,
        0.5f, 0.5f,
        -0.5f, 0.5f
    };

    ShaderInfo ground_plane_shader_info[] = {
        {GL_VERTEX_SHADER, "../shaders/ground.vs"},
        {GL_FRAGMENT_SHADER, "../shaders/ground.fs"},
        {GL_NONE, NULL}
    };

    GLuint ground_shader_program = LoadShaders(ground_plane_shader_info);

    printf("ground shader program: %d \n", ground_shader_program);

    unsigned int groundVAO, groundVBO;
    glGenVertexArrays(1, &groundVAO);
    glGenBuffers(1, &groundVBO);

    glBindVertexArray(groundVAO);
    glBindBuffer(GL_ARRAY_BUFFER, groundVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

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

        predict_sim_step();
        //genCollisionConstraints();
        solve();

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();

        //Display light cube
        glUseProgram(light_shader_program);

        glUniformMatrix4fv(glGetUniformLocation(light_shader_program, "projection"), 1, GL_FALSE, &projection[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(light_shader_program, "view"), 1, GL_FALSE, &view[0][0]);
        
        glBindVertexArray(lightVAO);
        
        for (int i = 0; i < num_fluid_particles; i++)
        {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::mat4(1.0f);
            //model = glm::translate(model, lightPos + glm::vec3(i,0,0));
            model = glm::translate(model, x[i]);
            model = glm::scale(model, glm::vec3(0.01f));
            if (i == 15)
            {
                //model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                glUniform1f(glGetUniformLocation(light_shader_program, "inCol"), 1.0f);

            } else {
                glUniform1f(glGetUniformLocation(light_shader_program, "inCol"), 0.5f);
            }

            glUniformMatrix4fv(glGetUniformLocation(light_shader_program, "model"), 1, GL_FALSE, &model[0][0]);

            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
       
        //End Display light cube

        // //Display ground plane
        // glUseProgram(ground_shader_program);

        // glUniformMatrix4fv(glGetUniformLocation(ground_shader_program, "projection"), 1, GL_FALSE, &projection[0][0]);
        // glUniformMatrix4fv(glGetUniformLocation(ground_shader_program, "view"), 1, GL_FALSE, &view[0][0]);
        
        // glm::mat4 model = glm::mat4(1.0f);
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
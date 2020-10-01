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

#include "FluidSimulator.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

const int SCREEN_WIDTH = 1280;
const int SCREEN_HEIGHT = 720;

// timing
bool debugMode = false;
float deltaTime = 0.0f;
float lastTime = 0.0f;
int numFrames = 0;
 
// camera
Camera camera(glm::vec3(-1.803095f,1.479963f,-2.037947f));

float lastX = SCREEN_WIDTH / 2.0f;
float lastY = SCREEN_HEIGHT / 2.0f;
bool firstMouse = true;

//simulation constants
int num_fluid_particles = 80000;
int gridWidth = 40;
float gridCellSize = 0.2f;
float dt = 0.0086f;

int main(void)
{
    if (!glfwInit())
        exit(EXIT_FAILURE);

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

    FluidSimulator *fs = new FluidSimulator(num_fluid_particles, gridCellSize, gridWidth);

    glEnable(GL_DEPTH_TEST);

    ShaderInfo fluid_shader_info[] = {
        {GL_VERTEX_SHADER, "../shaders/fluid.vs"},
        {GL_FRAGMENT_SHADER, "../shaders/fluid.fs"},
        {GL_NONE, NULL}
    };

    GLuint fluid_shader_program = LoadShaders(fluid_shader_info);

    uint cvbo = fs->getVBO();
    int cn = fs->getNumFluidParticles();
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

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 model = glm::mat4(1.0f);
        
        //Display light cube
        glUseProgram(fluid_shader_program);
        glEnable(GL_PROGRAM_POINT_SIZE);

        glUniformMatrix4fv(glGetUniformLocation(fluid_shader_program, "projection"), 1, GL_FALSE, &projection[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(fluid_shader_program, "view"), 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(fluid_shader_program, "model"), 1, GL_FALSE, &model[0][0]);

        glUniform3f(glGetUniformLocation(fluid_shader_program, "cameraCenter"), camera.Position[0], camera.Position[1], camera.Position[2]);

        glBindBuffer(GL_ARRAY_BUFFER, cvbo);
        glVertexPointer(3, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnable(GL_POINT_SPRITE_OES);
        glDrawArrays(GL_POINTS, 0, cn);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDisableClientState(GL_VERTEX_ARRAY);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    //fs->cleanUpSimulation();
    delete fs;

    glfwDestroyWindow(window);
 
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

void processInput(GLFWwindow *window)
{
    float offset = 0.01f;
    float dt = 0.01f;

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
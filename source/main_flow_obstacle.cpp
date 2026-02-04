#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "lbm_kernel.cuh"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

// ============================================================================
// Simulation Parameters
// ============================================================================

// Grid dimensions (wide aspect ratio for wake development)
const int GRID_WIDTH = 1600;
const int GRID_HEIGHT = 800;

// Window size
const int WINDOW_WIDTH = 1600;
const int WINDOW_HEIGHT = 800;

// Obstacle geometry
const int OBSTACLE_RADIUS = 120;  // Radius in lattice units
const int OBSTACLE_X = GRID_WIDTH / 5;
const int OBSTACLE_Y = GRID_HEIGHT / 2;


// Re = u * D / nu = u * 2R / ((tau - 0.5)/3)
// At tau = 0.56, nu = 0.02, D = 60, Re = 100 -> u = 100 * 0.02 / 60 = 0.033
const float TAU = 0.56f;
const float INLET_VELOCITY = 0.08f;  // Low Mach number for stability


GLuint shaderProgram;
GLuint VAO, VBO, EBO;
GLuint gridTexture;
cudaGraphicsResource* cudaResource = nullptr;


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
GLuint compileShader(const char* source, GLenum shaderType);
GLuint createShaderProgram(const char* vertexPath, const char* fragmentPath);
void setupQuad();
void setupTexture();
std::string readFile(const char* filepath);


std::string readFile(const char* filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

GLuint compileShader(const char* source, GLenum shaderType) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed:\n" << infoLog << std::endl;
    }

    return shader;
}

GLuint createShaderProgram(const char* vertexPath, const char* fragmentPath) {
    std::string vertexCode = readFile(vertexPath);
    std::string fragmentCode = readFile(fragmentPath);

    if (vertexCode.empty() || fragmentCode.empty()) {
        std::cerr << "Failed to load shaders!" << std::endl;
        return 0;
    }

    GLuint vertexShader = compileShader(vertexCode.c_str(), GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(fragmentCode.c_str(), GL_FRAGMENT_SHADER);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader linking failed:\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

void setupQuad() {
    float vertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    unsigned int indices[] = {
        0, 1, 2,
        2, 3, 0
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void setupTexture() {
    glGenTextures(1, &gridTexture);
    glBindTexture(GL_TEXTURE_2D, gridTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, GRID_WIDTH, GRID_HEIGHT, 0, GL_RGBA, GL_FLOAT, nullptr);

    glBindTexture(GL_TEXTURE_2D, 0);

    initCudaTexture(&cudaResource, gridTexture);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}


int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        WINDOW_WIDTH, WINDOW_HEIGHT,
        "LBM Flow Over Obstacle - Karman Vortex Street",
        nullptr, nullptr
    );

    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glfwSwapInterval(0);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    cudaSetDevice(0);
    cudaGLSetGLDevice(0);

    std::cout << "============================================" << std::endl;
    std::cout << "   Flow Over Obstacle Simulation" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Grid: " << GRID_WIDTH << " x " << GRID_HEIGHT << std::endl;
    std::cout << "Obstacle: radius " << OBSTACLE_RADIUS
              << ", center (" << OBSTACLE_X << ", " << OBSTACLE_Y << ")" << std::endl;

    float nu = (TAU - 0.5f) / 3.0f;
    float D = 2.0f * OBSTACLE_RADIUS;
    float Re = INLET_VELOCITY * D / nu;
    float Ma = INLET_VELOCITY / (1.0f / sqrtf(3.0f));

    std::cout << "Tau: " << TAU << std::endl;
    std::cout << "Viscosity nu: " << nu << std::endl;
    std::cout << "Inlet velocity: " << INLET_VELOCITY << std::endl;
    std::cout << "Reynolds number: " << Re << std::endl;
    std::cout << "Mach number: " << Ma << std::endl;
    std::cout << "============================================" << std::endl;

    if (Re > 40 && Re < 200) {
        std::cout << "Note: Re in range for Karman vortex shedding!" << std::endl;
    }

    shaderProgram = createShaderProgram("shaders/shader.vert", "shaders/shader.frag");
    if (shaderProgram == 0) {
        std::cerr << "Failed to create shader program" << std::endl;
        return -1;
    }

    setupQuad();
    setupTexture();

    LBM_State* lbmState = new LBM_State();
    initializeLBM_FlowObstacle(
        lbmState,
        GRID_WIDTH, GRID_HEIGHT,
        TAU,
        INLET_VELOCITY,
        OBSTACLE_X, OBSTACLE_Y, OBSTACLE_RADIUS
    );

    int frameCount = 0;
    float lastTime = glfwGetTime();
    int totalTimesteps = 0;

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        const int STEPS_PER_FRAME = 10;
        for (int i = 0; i < STEPS_PER_FRAME; i++) {
            stepLBM_FlowObstacle(cudaResource, lbmState);
            totalTimesteps++;
        }

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, gridTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "gridTexture"), 0);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();

        frameCount++;
        float currentTime = glfwGetTime();
        if (currentTime - lastTime >= 1.0f) {
            std::cout << "FPS: " << frameCount
                      << " | LBM steps/s: " << frameCount * STEPS_PER_FRAME
                      << " | Total timesteps: " << totalTimesteps << std::endl;
            frameCount = 0;
            lastTime = currentTime;
        }
    }


    lbm_destroy(lbmState);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteTextures(1, &gridTexture);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}

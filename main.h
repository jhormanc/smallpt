#ifndef MAIN_H
#define MAIN_H

#endif // MAIN_H

// GLM (vector / matrix)
#define GLM_FORCE_RADIANS
#include "glm/glm/vec4.hpp"
#include "glm/glm/vec3.hpp"
#include "glm/glm/mat4x4.hpp"
#include "glm/glm/gtc/matrix_transform.hpp"
#include "omp.h"
#include <ctime>

const float sigma = 0.3f; // Antialiasing
const float eps = 0.1f;
const glm::vec3 lux(1.f, 0.f, 0.f);

struct Ray
{
    const glm::vec3 origin, direction;
};

struct Sphere
{
    const float radius;
    const glm::vec3 center;
};

struct Triangle
{
    const glm::vec3 v0, v1, v2;
};

float random_u();
float V(glm::vec3 p, glm::vec3 l);
bool refract(glm::vec3 i, glm::vec3 n, float ior, glm::vec3 &wo);
glm::vec3 radiance (const Ray & r, int num_rebond);
glm::vec3 sample_cos(const float u, const float v, const glm::vec3 n);
float distance_squared(glm::vec3 a, glm::vec3 b);

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
const glm::vec3 eps(0.1f); // Ray direction epsilon
const glm::vec3 lux(1.f, 1.f, 1.f); // Lumière
const glm::vec3 I(10000.f); // Intensité lumineuse

// Nombre de rebonds max pour les appels récursifs à radiance() (dans indirect())
const int nb_rebonds = 5;

// Nombre de rayons lancés par pixel
const int nb_passages = 10;

// Indices de réfraction
const float n1 = 1.524f;
const float n2 = 1.f;

const float k0 = ((n1 - n2) * (n1 - n2)) / ((n1 + n2) * (n1 + n2));

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

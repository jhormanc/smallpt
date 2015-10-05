// This code is highly based on smallpt
// http://www.kevinbeason.com/smallpt/
#include "main.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <random>
#include <memory>
#include <fstream>
#include <iostream>
#include <iomanip>


thread_local std::default_random_engine generator;
thread_local std::uniform_real_distribution<float> distribution(0.0,1.0);

float random_u()
{
    return distribution(generator);
}

const float pi = 3.1415927f;
const float noIntersect = std::numeric_limits<float>::infinity();

bool isIntersect(float t)
{
    return t < noIntersect;
}

    // WARRING: works only if r.d is normalized
float intersect (const Ray & ray, const Sphere &sphere)
{				// returns distance, 0 if nohit
    glm::vec3 op = sphere.center - ray.origin;		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    float t, b = glm:: dot(ray.direction, op), det =
        b * b - glm::dot(op, op) + sphere.radius * sphere.radius;
    if (det < 0)
        return noIntersect;
    else
        det = std::sqrt (det);
    return (t = b - det) >= 0 ? t : ((t = b + det) >= 0 ? t : noIntersect);
}

float intersect(const Ray & ray, const Triangle &triangle)
{
    auto e1 = triangle.v1 - triangle.v0;
    auto e2 = triangle.v2 - triangle.v0;

    auto h = glm::cross(ray.direction, e2);
    auto a = glm::dot(e1, h);

    auto f = 1.f / a;
    auto s = ray.origin - triangle.v0;

    auto u = f * glm::dot(s, h);
    auto q = glm::cross(s, e1);
    auto v = f * glm::dot(ray.direction, q);
    auto t = f * glm::dot(e2, q);

    if(std::abs(a) < 0.00001)
        return noIntersect;
    if(u < 0 || u > 1)
        return noIntersect;
    if(v < 0 || (u+v) > 1)
        return noIntersect;
    if(t < 0)
        return noIntersect;

    return t;
}

/*Normal function*/
glm::vec3 normal(Sphere s, glm::vec3 p)
{
    return glm::normalize(p - s.center);
}

glm::vec3 normal(Triangle t ,glm::vec3 p)
{
    return glm::normalize(glm::cross((t.v1 - t.v0), (t.v2 - t.v0)));
}

struct Diffuse
{
    const glm::vec3 color;

    glm::vec3 direct(glm::vec3 c, glm::vec3 p, glm::vec3 n, glm::vec3 l) const
    {
        return V(p, l) * bsdf(c, n, glm::normalize(l - p));
    }

    glm::vec3 indirect(glm::vec3 c, glm::vec3 p, glm::vec3 np, int num_rebond) const
    {
        float u = random_u();
        float v = random_u();

        np = glm::normalize(np);
        float teta = glm::dot(glm::normalize(c), np);

        glm::vec3 w = sample_cos(u, v, teta < 0. ? -np : np);
        glm::vec3 pt = p + w * eps;
        Ray ray = Ray{pt, w};

        if(num_rebond > 0)
            return radiance(ray, num_rebond - 1) * color;
    }

    glm::vec3 bsdf(glm::vec3 c, glm::vec3 np, glm::vec3 l) const
    {
        return color * glm::abs(glm::dot(np, l)) / pi;
    }

    glm::vec3 brdf(glm::vec3 c, glm::vec3 dir)
    {
        return glm::vec3(1 / pi);
    }
};

struct Glass
{
    const glm::vec3 color;

    glm::vec3 direct(glm::vec3 c, glm::vec3 p, glm::vec3 n, glm::vec3 l) const
    {
        return V(p, l) * bsdf(c, n, glm::normalize(l - p));
    }

    glm::vec3 indirect(glm::vec3 c, glm::vec3 p, glm::vec3 np, int num_rebond) const
    {
        glm::vec3 n = glm::normalize(np);
        float ior = 1.33;
        glm::vec3 w;

        bool ref = refract(c, n, ior, w);

        if(ref)
        {
            glm::vec3 pt = p + w * eps;
            Ray ray = Ray{pt, w};

            if(num_rebond > 0)
                return radiance(ray, num_rebond - 1);
        }

        return glm::vec3(0.);
    }

    glm::vec3 bsdf(glm::vec3 c, glm::vec3 np, glm::vec3 l) const
    {
        return glm::vec3(0.);
    }

};

struct Mirror
{
    const glm::vec3 color;

    glm::vec3 direct(glm::vec3 c, glm::vec3 p, glm::vec3 n, glm::vec3 l) const
    {
        return V(p, l) * bsdf(c, n, glm::normalize(l - p));
    }

    glm::vec3 indirect(glm::vec3 c, glm::vec3 p, glm::vec3 np, int num_rebond) const
    {
        glm::vec3 eps(0.1);
        glm::vec3 r = reflect(-c, glm::normalize(np));
        glm::vec3 pt = p + r * eps;
        Ray ray = Ray{pt, r};

        if(num_rebond > 0)
            return radiance(ray, num_rebond - 1);

        return glm::vec3(0.);
    }

    glm::vec3 bsdf(glm::vec3 c, glm::vec3 np, glm::vec3 l) const
    {
        return glm::vec3(0.);
    }
};

template<typename T>
glm::vec3 albedo(const T &t)
{
    return t.color;
}

struct Object
{
    virtual float intersect(const Ray &r) const = 0;
    virtual glm::vec3 albedo() const = 0;
    virtual glm::vec3 normal(glm::vec3 p) const = 0;
    virtual glm::vec3 direct(glm::vec3 c, glm::vec3 p, glm::vec3 l) const = 0;
    virtual glm::vec3 indirect(glm::vec3 c, glm::vec3 p, int num_rebond) const = 0;
};

template<typename P, typename M>
struct ObjectTpl final : Object
{
    ObjectTpl(const P &_p, const M &_m)
        :primitive(_p), material(_m)
    {}

    float intersect(const Ray &ray) const
    {

        return ::intersect(ray, primitive);
    }

    glm::vec3 albedo() const
    {
        return ::albedo(material);
    }

    glm::vec3 normal(glm::vec3 p) const
    {
        return ::normal(primitive, p);
    }

    glm::vec3 direct(glm::vec3 c, glm::vec3 p, glm::vec3 l) const
    {
        return material.direct(c, p, normal(p), l);
    }

    glm::vec3 indirect(glm::vec3 c, glm::vec3 p, int num_rebond) const
    {
        return material.indirect(c, p, normal(p), num_rebond);
    }

    const P &primitive;
    const M &material;
};


template<typename P, typename M>
std::unique_ptr<Object> makeObject(const P&p, const M&m)
{
    return std::unique_ptr<Object>(new ObjectTpl<P, M>{p, m});
}

// Scene
namespace scene
{
    // Primitives

    // Left Wall
    const Triangle leftWallA{{0, 0, 0}, {0, 100, 0}, {0, 0, 150}};
    const Triangle leftWallB{{0, 100, 150}, {0, 100, 0}, {0, 0, 150}};

    // Right Wall
    const Triangle rightWallA{{100, 0, 0}, {100, 100, 0}, {100, 0, 150}};
    const Triangle rightWallB{{100, 100, 150}, {100, 100, 0}, {100, 0, 150}};

    // Back wall
    const Triangle backWallA{{0, 0, 0}, {100, 0, 0}, {100, 100, 0}};
    const Triangle backWallB{{0, 0, 0}, {0, 100, 0}, {100, 100, 0}};

    // Bottom Floor
    const Triangle bottomWallA{{0, 0, 0}, {100, 0, 0}, {100, 0, 150}};
    const Triangle bottomWallB{{0, 0, 0}, {0, 0, 150}, {100, 0, 150}};

    // Top Ceiling
    const Triangle topWallA{{0, 100, 0}, {100, 100, 0}, {0, 100, 150}};
    const Triangle topWallB{{100, 100, 150}, {100, 100, 0}, {0, 100, 150}};

    const Sphere leftSphere{16.5, glm::vec3 {27, 16.5, 47}};
    const Sphere rightSphere{16.5, glm::vec3 {73, 16.5, 78}};

    const glm::vec3 light{50, 70, 81.6};

    // Materials
    const Diffuse white{{.75, .75, .75}};
    const Diffuse red{{.75, .25, .25}};
    const Diffuse blue{{.25, .25, .75}};

    const Glass glass{{.9, .1, .9}};
    const Mirror mirror{{.9, .9, .1}};

    // Objects
    // Note: this is a rather convoluted way of initialising a vector of unique_ptr ;)
    const std::vector<std::unique_ptr<Object>> objects = [] (){
        std::vector<std::unique_ptr<Object>> ret;
        ret.push_back(makeObject(backWallA, white));
        ret.push_back(makeObject(backWallB, white));
        ret.push_back(makeObject(topWallA, white));
        ret.push_back(makeObject(topWallB, white));
        ret.push_back(makeObject(bottomWallA, white));
        ret.push_back(makeObject(bottomWallB, white));
        ret.push_back(makeObject(rightWallA, blue));
        ret.push_back(makeObject(rightWallB, blue));
        ret.push_back(makeObject(leftWallA, red));
        ret.push_back(makeObject(leftWallB, red));

        ret.push_back(makeObject(leftSphere, mirror));
        ret.push_back(makeObject(rightSphere, glass));

        return ret;
    }();
}

glm::vec3 sample_cos(const float u, const float v, const glm::vec3 n)
{
    // Ugly: create an ornthogonal base
    glm::vec3 basex, basey, basez;

    basez = n;
    basey = glm::vec3(n.y, n.z, n.x);

    basex = glm::cross(basez, basey);
    basex = glm::normalize(basex);

    basey = glm::cross(basez, basex);

    // cosinus sampling. Pdf = cosinus
    return  basex * (std::cos(2.f * pi * u) * std::sqrt(1.f - v)) +
        basey * (std::sin(2.f * pi * u) * std::sqrt(1.f - v)) +
        basez * std::sqrt(v);
}

int toInt (const float x)
{
    return int (std::pow (glm::clamp (x, 0.f, 1.f), 1.f / 2.2f) * 255 + .5);
}

// WARNING: ASSUME NORMALIZED RAY
// Compute the intersection ray / scene.
// Returns true if intersection
// t is defined as the abscisce along the ray (i.e
//             p = r.o + t * r.d
// id is the id of the intersected object
Object* intersect (const Ray & r, float &t)
{
    t = noIntersect;
    Object *ret = nullptr;

    for(auto &object : scene::objects)
    {
        float d = object->intersect(r);
        if (isIntersect(d) && d < t)
        {
            t = d;
            ret = object.get();
        }
    }

    return ret;
}

// renvoie 1 si le point p n'a aucun d'objet entre p et l, sinon 0
float V(glm::vec3 p, glm::vec3 l)
{
    float t;
    float eps = 0.1;

    // vecteur directeur entre le point d'intersection et la lumière
    glm::vec3 dir = glm::normalize(l - p);

    // on déplace le point p de epsilon dans la direction de la lumière
    glm::vec3 pt = p + dir * eps;

    Ray ray = Ray{pt, dir};
    intersect(ray, t);

    // pas d'intersection entre le point et la source de lumière ou si l'objet obj2 est situé à une distance t2 plus grande que d
    if (!isIntersect(2) || t > (float)glm::distance(pt, l))
        return 1;

    return 0;
}

// Reflect the ray i along the normal.
// i should be oriented as "leaving the surface"
glm::vec3 reflect(const glm::vec3 i, const glm::vec3 n)
{
    return n * (glm::dot(n, i)) * 2.f - i;
}

float sin2cos (const float x)
{
    return std::sqrt(std::max(0.0f, 1.0f-x*x));
}

// Fresnel coeficient of transmission.
// Normal point outside the surface
// ior is n0 / n1 where n0 is inside and n1 is outside
float fresnelR(const glm::vec3 i, const glm::vec3 n, const float ior)
{
    if(glm::dot(n, i) < 0)
        return fresnelR(i, n * -1.f, 1.f / ior);

    float R0 = (ior - 1.f) / (ior + 1.f);
    R0 *= R0;

    return R0 + (1.f - R0) * std::pow(1.f - glm::dot(i, n), 5.f);
}

// compute refraction vector.
// return true if refraction is possible.
// i and n are normalized
// output wo, the refracted vector (normalized)
// n point oitside the surface.
// ior is n00 / n1 where n0 is inside and n1 is outside
//
// i point outside of the surface
bool refract(glm::vec3 i, glm::vec3 n, float ior, glm::vec3 &wo)
{
    i = i * -1.f;

    if(glm::dot(n, i) > 0)
    {
        n = n * -1.f;
    }
    else
    {
        ior = 1.f / ior;
    }

    float k = 1.f - ior * ior * (1.f - glm::dot(n, i) * glm::dot(n, i));
    if (k < 0.)
        return false;

    wo = i * ior - n * (ior * glm::dot(n, i) + std::sqrt(k));

    return true;
}

glm::vec3 sample_sphere(const float r, const float u, const float v, float &pdf, const glm::vec3 normal)
{
    pdf = 1.f / (pi * r * r);
    glm::vec3 sample_p = sample_cos(u, v, normal);

    float cos = glm::dot(sample_p, normal);

    pdf *= cos;
    return sample_p * r;
}

glm::vec3 radiance (const Ray & r, int num_rebond)
{
    int nb_echantillons = 100;

    for(int i = 0; i < nb_echantillons; i++)
    {
        float t;
        Object* obj = intersect(r, t);

        if(isIntersect(t))
        {
            glm::vec3 p = r.origin + r.direction * t; // point d'intersection du rayon r;
            glm::vec3 lum = scene::light;
            glm::vec3 cam = glm::normalize(r.origin - p);

            return obj->direct(cam, p, lum) + obj->indirect(cam, p, num_rebond);
        }

        return glm::vec3(0.0f);
    }
}

int main (int, char **)
{
    int w = 768, h = 768;
    std::vector<glm::vec3> colors(w * h, glm::vec3{0.f, 0.f, 0.f});

    Ray cam {{50, 52, 295.6}, glm::normalize(glm::vec3{0, -0.042612, -1})};	// cam pos, dir
    float near = 1.f;
    float far = 10000.f;

    glm::mat4 camera =
        glm::scale(glm::mat4(1.f), glm::vec3(float(w), float(h), 1.f))
        * glm::translate(glm::mat4(1.f), glm::vec3(0.5, 0.5, 0.f))
        * glm::perspective(float(54.5f * pi / 180.f), float(w) / float(h), near, far)
        * glm::lookAt(cam.origin, cam.origin + cam.direction, glm::vec3(0, 1, 0))
        ;

    glm::mat4 screenToRay = glm::inverse(camera);

    // Nombre de rebonds max pour les appels récursifs à radiance() (dans indirect())
    int nb_rebonds = 15;

    // Nombre de rayons lancés par pixel
    int nb_passages = 100;

    for (int y = 0; y < h; y++)
    {
        std::cerr << "\rRendering: " << 100 * y / (h - 1) << "%";

        for (unsigned short x = 0; x < w; x++)
        {
            glm::vec3 sum_r = glm::vec3(0.f);
            //#pragma omp parallel for schedule(dynamic, 1)
            for(unsigned short k = 0; k < nb_passages; k++)
            {
                glm::vec4 p0 = screenToRay * glm::vec4{float(x), float(h - y), 0.f, 1.f};
                glm::vec4 p1 = screenToRay * glm::vec4{float(x), float(h - y), 1.f, 1.f};

                glm::vec3 pp0 = glm::vec3(p0 / p0.w);
                glm::vec3 pp1 = glm::vec3(p1 / p1.w);

                glm::vec3 d = glm::normalize(pp1 - pp0);

                glm::vec3 r = radiance (Ray{pp0, d}, nb_rebonds);

                sum_r += r;
                //colors[y * w + x] += glm::clamp(r, glm::vec3(0.f, 0.f, 0.f), glm::vec3(1.f, 1.f, 1.f));
            }

            //colors[y * w + x] /= glm::vec3(nb_passages);
            colors[y * w + x] = glm::clamp(sum_r / glm::vec3(nb_passages), glm::vec3(0.f, 0.f, 0.f), glm::vec3(1.f, 1.f, 1.f));
        }
    }

    {
        std::fstream f("image.ppm", std::fstream::out);
        f << "P3\n" << w << " " << h << std::endl << "255" << std::endl;

        for (auto c : colors)
            f << toInt(c.x) << " " << toInt(c.y) << " " << toInt(c.z) << " ";
    }
}

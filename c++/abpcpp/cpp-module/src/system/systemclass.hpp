#ifndef __SYSTEMCLASS_HPP__
#define __SYSTEMCLASS_HPP__

#include "particletype.hpp"
#include "../box/box.hpp"

class SystemClass
{
    public:
    /**
     * @brief Create an empty particles list
     * @param void
    */
    SystemClass(const BoxType &box) : _box(box), Numparticles(0) {particles.clear();}
    /**
     * Create a particles from given vectors
    */
    SystemClass(const host::vector<ParticleType> &_particles, const BoxType &box): _box(box)
    {
        this->set(_particles);
    }
    ~SystemClass() {}

    /**
     * @brief get the particles loaded in the system
     * @param void 
     * @return const BoxType&
     */
    const BoxType& get_box(void) {return _box;}

    /**
     * @brief get the particles loaded in the system
     * @param void 
     * @return host::vector<ParticleType> 
     */
    host::vector<ParticleType> get(void) { return (particles); }

    /**
     * @brief get the particles loaded in the system
     * @param const host::vector<ParticleType>& _particles 
     * @return void
     */
    void set(const host::vector<ParticleType> &_particles)
    {
        particles = _particles;
        Numparticles = _particles.size();
    }
    /**
     * @brief add a single particle into the system
     * @param ParticleType&
     * @return void
     */
    void add_particle(ParticleType &_particle)
    {
        _particle.id = Numparticles;
        particles.push_back(_particle);
        Numparticles = particles.size();
    }

    /** Variables **/
    host::vector<ParticleType> particles; //!< Particle list
    int Numparticles;                     //!< Number of particles

    private:
    BoxType _box;

    

};

#endif
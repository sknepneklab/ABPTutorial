#include <algorithm>
#include "evolverclass.hpp"

//operators over the particles
#include "../system/particleoperators_dev.hpp"

//here include all the hpp files of the forces
#include "../potentials/harmonicforce.hpp"
#include "../potentials/selfpropulsion.hpp"

//here include all the hpp files of the torques
#include "../potentials/polar_align.hpp"

//here include all the hpp files of the integrators
#include "../integrators/integrator_brownian_positions.hpp"
#include "../integrators/integrator_brownian_rotational.hpp"

//neighbour list
std::map<std::string, host::vector<int>> EvolverClass::get_neighbourlist(void)
{
    return(neighbourlist->get_neighbourlist());
}

void EvolverClass::alloc_neighbourlist(void)
{
    neighbourlist = std::make_shared<NeighbourListType>(_system);
    this->create_neighbourlist(1.0);
}
void EvolverClass::create_neighbourlist(const real &rcut)
{
    neighbourlist->create_neighbourlist(rcut);
    this->fill_neighbourlist();
}
void EvolverClass::fill_neighbourlist(void)
{
    neighbourlist->fill_neighbourlist();
}
void EvolverClass::update_neighbourlist(void)
{
    //retrieve the rcut from all the forces
    for (const auto& flist : force_list)
        rcut_list["force " + flist.first] = flist.second->get_rcut();
    //retrieve the rcut from all the torques
    for (const auto& tlist : torque_list)
        rcut_list["torques " + tlist.first] = tlist.second->get_rcut();
    //find the maximum rcut
    /* 
        this a rather advance way to find the maximum element in map values
        the idea is use std::max_element alongside a lambda function that retrieve and compare the values (no keys) in the map
        see: https://en.cppreference.com/w/cpp/algorithm/max_element
             https://en.cppreference.com/w/cpp/language/lambda
        the result value is a std::map containing the key and value in map values
    */
    auto max_it = std::max_element(rcut_list.begin(), rcut_list.end(), [](const std::pair<std::string, real> &p1, const std::pair<std::string, real> &p2) { return p1.second < p2.second; });
    //create/update the neighbour list
    if (max_it->second > 0.0)
        this->create_neighbourlist(max_it->second);
}

//potentials and torques
void EvolverClass::add_force(const std::string &name, std::map<std::string, real> &parameters)
{
    if (name.compare("Harmonic Force") == 0)
    {
        //add the force to the list
        force_list[name] = std::make_unique<HarmonicForce>(_system, *neighbourlist.get());
        //loop over the parameters and set them up
        for (auto param : parameters)
            force_list[name]->set_property(param.first, param.second);
        //if this potential change the global rcut the neighbourlist must be updated
        this->update_neighbourlist();
    }
    else if (name.compare("Self Propulsion") == 0)
    {
        //add the force to the list
        force_list[name] = std::make_unique<SelfPropulsionForce>(_system, *neighbourlist.get());
        for (auto param : parameters)
            force_list[name]->set_property(param.first, param.second);
    }
    else
        std::cerr << name << " potential not found" << std::endl;
}
void EvolverClass::add_torque(const std::string &name, std::map<std::string, real> &parameters)
{
    if (name.compare("Polar Align") == 0)
    {
        //add the force to the list
        torque_list[name] = std::make_unique<PolarAlign>(_system, *neighbourlist.get());
        //loop over the parameters and set them up
        for (auto param : parameters)
            torque_list[name]->set_property(param.first, param.second);
        //if this potential change the global rcut the neighbourlist must be updated
        this->update_neighbourlist();
    }
    else
        std::cerr << name << " torque not found" << std::endl;
}
//compute
void EvolverClass::reset_forces_torques_energy(void)
{
    // Note: one of advantages of using classes is that we can treat them as "objects" with define operations
    // for example here we have subtitude a for loop by a transformation over the particle list.
    // Check  "particleoperators.hpp"
    // More complex behaviour can be achieved by overloading operators: https://en.cppreference.com/w/cpp/language/operators
    thrust::transform(_system.particles.begin(), _system.particles.end(), _system.particles.begin(), device::reset_particle_forces_torques_energy());
}

void EvolverClass::reset_forces(void)
{
    thrust::transform(_system.particles.begin(), _system.particles.end(), _system.particles.begin(), device::reset_particle_forces());
}

void EvolverClass::compute_forces(void)
{
    for (const auto& force : force_list)
        force.second->compute();
}

void EvolverClass::reset_torques(void)
{
    thrust::transform(_system.particles.begin(), _system.particles.end(), _system.particles.begin(), device::reset_particle_torques());
}

void EvolverClass::compute_torques(void)
{
    for (const auto& torque : torque_list)
        torque.second->compute();
}

void EvolverClass::reset_energy(void)
{
    thrust::transform(_system.particles.begin(), _system.particles.end(), _system.particles.begin(), device::reset_particle_energy());
}

void EvolverClass::compute_energy(void)
{
    neighbourlist->automatic_update();
    for (const auto& force : force_list)
        force.second->compute_energy();
    for (const auto& torque : torque_list)
        torque.second->compute_energy();
}

void EvolverClass::add_integrator(const std::string &name, std::map<std::string, real> &parameters)
{
    //add the integrator to the list
    if (name.compare("Brownian Positions") == 0)
    {
        integrator_list[name] = std::make_shared<IntegratorBrownianParticlesPositions>(_system);
        for (auto param : parameters)
            integrator_list[name]->set_property(param.first, param.second);
    }
    else if (name.compare("Brownian Rotation") == 0)
    {
        integrator_list[name] = std::make_shared<IntegratorBrownianParticlesRotational>(_system);
        for (auto param : parameters)
            integrator_list[name]->set_property(param.first, param.second);
    }
    else
        std::cerr << name << " integrator not found" << std::endl;
}

void EvolverClass::set_time_step(const real &dt)
{
    for (auto integrator : integrator_list)
        integrator.second->set_time_step(dt);
}

void EvolverClass::set_global_temperature(const real &T)
{
    for (auto integrator : integrator_list)
        integrator.second->set_temperature(T);
}

void EvolverClass::evolve(void)
{
    // Check is neighbour list needs rebuilding
    neighbourlist->automatic_update();

    // Perform the preintegration step, i.e., step before forces and torques are computed
    for (auto integrator : integrator_list)
        integrator.second->prestep();

    // Apply period boundary conditions
    _system.apply_periodic();

    // Reset all forces and toques
    this->reset_forces_torques_energy();

    // Compute all forces and torques
    this->compute_forces();
    this->compute_torques();

    // Perform the second step of integration
    for (auto integrator : integrator_list)
        integrator.second->poststep();

    // Apply period boundary conditions
    _system.apply_periodic();
}
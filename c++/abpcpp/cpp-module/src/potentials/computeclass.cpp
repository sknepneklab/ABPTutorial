#include <algorithm>
#include "computeclass.hpp"

//operators over the particles
#include "../system/particleoperators.hpp"

//here include all the hpp files of the forces
#include "harmonicforce.hpp"
//here include all the hpp files of the torques
#include "polar_align.hpp"

//neighbour list
void ComputeClass::alloc_neighbourlist(void)
{
    neighbourlist = std::make_shared<NeighbourListType>(_system);
}
void ComputeClass::create_neighbourlist(const real &rcut)
{
    neighbourlist->create_neighbourlist(rcut);
    this->fill_neighbourlist();
}
void ComputeClass::fill_neighbourlist(void)
{
    this->fill_neighbourlist();
}
void ComputeClass::update_neighbourlist(void)
{
    //retrieve the rcut from all the forces
    for (auto flist : force_list)
        rcut_list["force " + flist.first] = flist.second->get_rcut();
    //retrieve the rcut from all the torques
    for (auto tlist : torque_list)
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
    //create/update the nriubour list
    if (max_it->second > 0.0)
        this->create_neighbourlist(max_it->second);
}

//potentials and torques
void ComputeClass::add_force(const std::string &name, std::map<std::string, real> &parameters)
{
    if (name.compare("Harmonic Force") == 0)
    {
        //add the force to the list
        force_list[name] = std::make_shared<ComputeHarmonicForce>(_system, *neighbourlist.get());
        //loop over the parameters and set them up
        for (auto param : parameters)
            force_list[name]->set_property(param.first, param.second);
        //if this potential change the global rcut the neighbourlist must be updated
        this->update_neighbourlist();
    }
    else
        std::cerr << name << " potential not found" << std::endl;
}
void ComputeClass::add_torque(const std::string &name, std::map<std::string, real> &parameters)
{
    if (name.compare("Polar Align") == 0)
    {
        //add the force to the list
        torque_list[name] = std::make_shared<PolarAlign>(_system, *neighbourlist.get());
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
void ComputeClass::reset_forces_torques_energy(void)
{
    // Note: one of advantages of using classes is that we can treat them as "objects" with define operations
    // for example here we have subtitude a for loop by a transformation over the particle list.
    // Check  "particleoperators.hpp"
    // More complex behaviour can be achieved by overloading operators: https://en.cppreference.com/w/cpp/language/operators
    std::transform(_system.particles.begin(), _system.particles.end(), _system.particles.begin(), reset_particle_forces_torques_energy());
}

void ComputeClass::reset_forces(void)
{
    std::transform(_system.particles.begin(), _system.particles.end(), _system.particles.begin(), reset_particle_forces());
}

void ComputeClass::compute_forces(void)
{
    neighbourlist->automatic_update();
    for (auto f : force_list)
        f.second->compute();
}

void ComputeClass::reset_torques(void)
{
    std::transform(_system.particles.begin(), _system.particles.end(), _system.particles.begin(), reset_particle_torques());
}

void ComputeClass::compute_torque(void) 
{
    neighbourlist->automatic_update();
    for (auto f : torque_list)
        f.second->compute();
}

void ComputeClass::reset_energy(void)
{
    std::transform(_system.particles.begin(), _system.particles.end(), _system.particles.begin(), reset_particle_energy());
}

void ComputeClass::compute_energy(void) 
{
    neighbourlist->automatic_update();
    for (auto f : force_list)
        f.second->compute_energy();
    for (auto f : torque_list)
        f.second->compute_energy();
}

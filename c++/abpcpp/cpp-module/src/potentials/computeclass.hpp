#ifndef __computeclass_HPP__
#define __computeclass_HPP__

#include <memory>
#include <map>
#include <iostream>

#include "../neighbourlist/neighbourlistclass.hpp"
#include "../system/systemclass.hpp"
#include "computeforceclass.hpp"
#include "computetorqueclass.hpp"
#include "../neighbourlist/neighbourlistclass.hpp"

class ComputeClass
{
public:
    ComputeClass(SystemClass &system) : _system(system)
    {
        this->alloc_neighbourlist();
    }
    ~ComputeClass() {}

    //neighbour list
    void alloc_neighbourlist(void);
    void create_neighbourlist(const real&);
    void fill_neighbourlist(void);
    void update_neighbourlist(void);

    //potentials and torques
    void add_force(const std::string &name, std::map<std::string, real> & parameters);
    void add_torque(const std::string &name, std::map<std::string, real> & parameters);

    //compute
    void reset_forces_torques_energy(void);
    
    void reset_forces(void);
    void compute_forces(void);

    void reset_torques(void);
    void compute_torque(void);

    void reset_energy(void);
    void compute_energy(void);

private:
    SystemClass& _system;                                       //!< reference to system class where the box and particles are stored
    NeighbourListType_ptr neighbourlist;                         //!< neighbour list used for the force/torque calculation
    std::map<std::string, real> rcut_list;                      //!< list of all the rcut defined in the forces and torques
    std::map<std::string, ComputeForceClass_ptr> force_list;    //!< list of all the pointer to the forces
    std::map<std::string, ComputeTorqueClass_ptr> torque_list;  //!< list of all the pointer to the torques
};

#endif

/** @} */

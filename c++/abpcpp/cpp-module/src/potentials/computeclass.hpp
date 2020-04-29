#ifndef __computeclass_HPP__
#define __computeclass_HPP__

#include <memory>
#include <map>
#include <iostream>

#include "../system/systemclass.hpp"
#include "computeforceclass.hpp"
#include "computetorqueclass.hpp"
#include "../neighbourlist/neighbourlistclass.hpp"

class ComputeClass
{
public:
    ComputeClass(SystemClass &system) : _system(system)
    {
    }
    ~ComputeClass() {}


    //potentials and torques
    void add_force(const std::string &name, std::map<std::string, real> & parameters){ }
    void add_torque(const std::string &name, std::map<std::string, real> & parameters){ }




    //compute
    void compute_forces(void){ }
    void compute_torque(void){ }

private:
    SystemClass& _system;
};

#endif

/** @} */

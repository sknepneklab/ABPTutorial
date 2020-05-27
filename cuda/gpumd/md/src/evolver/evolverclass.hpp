/************************************************************************************
* MIT License                                                                       *
*                                                                                   *
* Copyright (c) 2020 Dr. Daniel Alejandro Matoz Fernandez                           *
*               fdamatoz@gmail.com                                                  *
* Permission is hereby granted, free of charge, to any person obtaining a copy      *
* of this software and associated documentation files (the "Software"), to deal     *
* in the Software without restriction, including without limitation the rights      *
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         *
* copies of the Software, and to permit persons to whom the Software is             *
* furnished to do so, subject to the following conditions:                          *
*                                                                                   *
* The above copyright notice and this permission notice shall be included in all    *
* copies or substantial portions of the Software.                                   *
*                                                                                   *
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        *
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          *
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       *
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            *
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     *
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     *
* SOFTWARE.                                                                         *
*************************************************************************************/
#ifndef __evolverclass_hpp__
#define __evolverclass_hpp__

#include <memory>
#include <map>
#include <iostream>

#include "../neighbourlist/neighbourlistclass.hpp"
#include "../system/systemclass.hpp"
#include "../potentials/computeforceclass.hpp"
#include "../potentials/computetorqueclass.hpp"
#include "../integrators/integratorclass.hpp"

class EvolverClass
{
public:
    EvolverClass(SystemClass &system) : _system(system)
    {
        this->alloc_neighbourlist();
    }
    ~EvolverClass() {}

    //neighbour list
    void alloc_neighbourlist(void);
    void create_neighbourlist(const real&);
    void fill_neighbourlist(void);
    void update_neighbourlist(void);

    //potentials and torques
    void add_force(const std::string &, std::map<std::string, real> & );
    void add_torque(const std::string &, std::map<std::string, real> & );

    //compute potentials
    void reset_forces_torques_energy(void);
    
    void reset_forces(void);
    void compute_forces(void);

    void reset_torques(void);
    void compute_torques(void);

    void reset_energy(void);
    void compute_energy(void);

    //Integrators
    void add_integrator(const std::string &, std::map<std::string, real> &);

    //Evolve
    void set_time_step(const real&);
    void set_global_temperature(const real&);
    void evolve(void);

    std::map<std::string, host::vector<int>> get_neighbourlist(void);

private:
    SystemClass& _system;                                       //!< reference to system class where the box and particles are stored
    NeighbourListType_ptr neighbourlist;                         //!< neighbour list used for the force/torque calculation
    std::map<std::string, real> rcut_list;                      //!< list of all the rcut defined in the forces and torques
    std::map<std::string, ComputeForceClass_ptr> force_list;    //!< list of all the pointer to the forces
    std::map<std::string, ComputeTorqueClass_ptr> torque_list;  //!< list of all the pointer to the torques
    std::map<std::string, IntegratorClass_ptr> integrator_list;  //!< list of all the pointer to the torques
};

#endif

/** @} */

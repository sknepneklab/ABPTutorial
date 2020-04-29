#ifndef __harmonicforce_HPP__
#define __harmonicforce_HPP__

#include <memory>
#include <map>
#include <iostream>

#include "computeforceclass.hpp"
#include "../neighbourlist/neighbourlistclass.hpp"

class ComputeHarmonicForce : ComputeForceClass
{
public:
    ComputeHarmonicForce(SystemClass &system, NeighbourListType &neighbourslist) : _neighbourslist(neighbourslist), ComputeForceClass(system)
    {
        name = "Harmonic Force";
        type = "Conservative/Particle";
        this->set_defaults_property();
    }
    ~ComputeHarmonicForce() {}

    void set_defaults_property(void)
    {
        k = 1.0;
        a = 2.0;
        rcut = a; //cut off radius to be use in neighbourslist
    }

    void set_property(const std::string &name, const double &value)
    {
        if (name.compare("k"))
            k = value;
        else if (name.compare("a"))
        {
            a = 2.0;
            rcut = a; //cut off radius to be use in neighbourslist
        }
        else
            this->print_warning_property_name(name);
    }
    
    void compute_energy(void);
    
    void compute(void);

private:
    NeighbourListType &_neighbourslist;
    real k, a;
};

#endif

/** @} */

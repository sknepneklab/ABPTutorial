#ifndef __polar_align_HPP__
#define __polar_align_HPP__

#include <memory>
#include <map>
#include <iostream>

#include "computetorqueclass.hpp"
#include "../neighbourlist/neighbourlistclass.hpp"

class PolarAlign : public ComputeTorqueClass
{
public:
    PolarAlign(SystemClass &system, NeighbourListType &neighbourslist) : _neighbourslist(neighbourslist), ComputeTorqueClass(system)
    {
        name = "Polar Align";
        type = "Torque/Particle";
        this->set_defaults_property();
    }
    ~PolarAlign() {}

    void set_defaults_property(void)
    {
        J = 1.0;
        a = 2.0;
        rcut = a; //cut off radius to be use in neighbourslist
    }

    void set_property(const std::string &name, const double &value)
    {
        if (name.compare("k"))
            J = value;
        else if (name.compare("a"))
        {
            a = value;
            rcut = a; //cut off radius to be use in neighbourslist
        }
        else
            this->print_warning_property_name(name);
    }
    
    void compute_energy(void);
    void compute(void);

private:
    NeighbourListType &_neighbourslist;
    real J, a;
};

#endif

/** @} */

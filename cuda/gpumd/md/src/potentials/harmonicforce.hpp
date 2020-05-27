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
#ifndef __harmonicforce_hpp__
#define __harmonicforce_hpp__

#include <memory>
#include <map>
#include <iostream>

#include "computeforceclass.hpp"
#include "../neighbourlist/neighbourlistclass.hpp"

class HarmonicForce : public ComputeForceClass
{
public:
    HarmonicForce(SystemClass &system, NeighbourListType &neighbourslist) : _neighbourslist(neighbourslist), ComputeForceClass(system)
    {
        name = "Harmonic Force";
        type = "Conservative/Particle";
        this->set_default_properties();
    }
    ~HarmonicForce() {}

    void set_default_properties(void) override
    {
        k = 1.0;
        a = 2.0;
        a2 = a*a;
        rcut = a; //cut off radius to be use in neighbourslist
    }

    void set_property(const std::string &prop_name, const double &value) override
    {
        if (prop_name.compare("k")==0)
            k = value;
        else if (prop_name.compare("a")==0)
        {
            a = value;
            a2 = a*a;
            rcut = a; //cut off radius to be use in neighbourslist
        }
        else
            this->print_warning_property_name(prop_name);
    }
    
    void compute_energy(void) override;
    
    void compute(void) override;

private:
    NeighbourListType &_neighbourslist;
    real k, a;
    real a2;
};

#endif

/** @} */

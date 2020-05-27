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
#ifndef __neighbourlistclass_hpp__
#define __neighbourlistclass_hpp__

#include <memory>
#include "linkedlistclass.hpp"
#include "../box/pbc.hpp"

struct NeighbourListType : public LinkedListType
{
    NeighbourListType(SystemClass &system) : LinkedListType(system)
    {
    }
    ~NeighbourListType() {}
    //linkedlist functions
    void create_neighbourlist(real _rcut, real _skin = 0.3, int _max_ng_per_particle = 100)
    {
        skin = _skin;
        skin2 = skin*skin;
        rcut = (1.0+skin)*_rcut;
        rcut2 = rcut*rcut;
        max_ng_per_particle = _max_ng_per_particle;
        this->create_linkedlist(rcut);
        host::vector<int> _nglist(max_ng_per_particle*_system.Numparticles);
        nglist = _nglist;
        host::vector<real2> _old_positions(_system.Numparticles+1);
        real2 value;
        value.x = value.y = 0.0;
        _old_positions[_system.Numparticles] = value;
        old_positions = _old_positions;
        //old_positions.resize(_system.Numparticles+1);
        //nglist.resize(max_ng_per_particle*_system.Numparticles);
        std::cout << " ********************************************\n";
        std::cout << " Neighbourlist list\n";
        std::cout << "neighbourlist.size()=" << nglist.size() << "\n";
        std::cout << "neighbourlist rcut =" << rcut << "\n";
        std::cout << "neighbourlist skin =" << skin << "\n";
        std::cout << " ********************************************\n";
    }
    void fill_neighbourlist(void);
    void fill_neighbourlist_brute_force(void);
    void automatic_update(void);
    std::map<std::string, std::vector<int>> get_neighbourlist(void);

    //Variables
    real rcut, rcut2;
    real skin,skin2;
    int max_ng_per_particle;
    device::vector<int> nglist;    ///< stores the neibourlist for the cells   
    device::vector<real2> old_positions;  ///< stores the particle position at the time of neibourlist creation

};

typedef std::shared_ptr<NeighbourListType> NeighbourListType_ptr;

#endif
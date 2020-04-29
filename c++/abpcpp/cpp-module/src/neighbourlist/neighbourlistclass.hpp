#ifndef __neighbourlistclass_HPP__
#define __neighbourlistclass_HPP__

#include <memory>
#include "linkedlistclass.hpp"
#include "../box/pbc.hpp"

struct NeighbourListType : LinkedListType
{
    NeighbourListType(SystemClass &system) : LinkedListType(system)
    {
    }
    ~NeighbourListType() {}
    //linkedlist functions
    void create_neighbourlist(real _rcut, real _skin = 0.3, int _max_ng_per_particle = 100)
    {
        rcut = (1.0+skin)*_rcut;
        rcut2 = rcut*rcut;
        skin = _skin;
        skin2 = skin*skin;
        max_ng_per_particle = 100;
        this->create_linkedlist(rcut);
        neighbourlist.resize(max_ng_per_particle*Numparticles);
    }
    void fill_neighbourlist(void);
    host::vector<int> get_neighbourlist(void);

    //Variables
    real rcut, rcut2;
    real skin,skin2;
    int max_ng_per_particle;
    host::vector<int> neighbourlist;    ///< stores the neibourlist for the cells   
    host::vector<real3> old_positions;  ///< stores the particle position at the time of neibourlist creation   
};

typedef std::shared_ptr<NeighbourListType> NeighbourListType_ptr;

#endif
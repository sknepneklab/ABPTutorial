#ifndef __neighbourlistclass_HPP__
#define __neighbourlistclass_HPP__

#include "linkedlistclass.hpp"

struct neighbourlistType: LinkedListType
{
    //public:
    neighbourlistType(host::vector<ParticleType> &particles, BoxType &box) : LinkedListType(particles, box)
    {
    }
    ~neighbourlistType() {}
    //linkedlist functions
    void create_neighbourlist(real _rcut, real skin=0.3)
    {

    }
    void reset_neighbourlist(void);
    void fill_neighbourlist(void);
    std::map<std::string, std::vector<int>> get_neighbourlist(void);

public:
    int max_num_neighbours;
    host::vector<int> neighbourlist; ///< stores the neibourlist for the cells    device::vector<int> neighbourlist;
};

void neighbourlistType::reset_neighbourlist(void)
{

}

void neighbourlistType::fill_neighbourlist(void)
{
    //reset the linked list
    this->reset_neighbourlist();

}

std::map<std::string, std::vector<int>> neighbourlistType::get_neighbourlist(void)
{

}

#endif
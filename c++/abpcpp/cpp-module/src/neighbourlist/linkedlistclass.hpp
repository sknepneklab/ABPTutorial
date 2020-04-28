#ifndef __linkedlistclass_HPP__
#define __linkedlistclass_HPP__

#include <iostream>
#include <map>
#include "../types/globaltypes.hpp"
#include "../particle/particles.hpp"
#include "../box/box.hpp"

class LinkedListType
{
    public:
    LinkedListType(host::vector<ParticleType> &particles, BoxType &box) : _particles(particles), _box(box)
    {
    }
    ~LinkedListType() {}
    //linkedlist functions
    void create_linkedlist(real _lcut)
    {
        lcut = _lcut;
        lcut2 = lcut * lcut;
        Ncells.x = ((int)(_box.L.x / (lcut)));      ///< Numbers of cells x
        Ncells.y = ((int)(_box.L.y / (lcut)));      ///< Numbers of cells y
        Ncells.z = ((int)(_box.L.z / (lcut)));      ///< Numbers of cells z
        LengthCells.x = (real)(_box.L.x / Ncells.x); ///< Length of the cells x
        LengthCells.y = (real)(_box.L.y / Ncells.y); ///< Length of the cells y
        LengthCells.z = (real)(_box.L.z / Ncells.z); ///< Length of the cells z
        NumCells = Ncells.x * Ncells.y * Ncells.z;
        if (NumCells > 0)
        {
            cellHead.clear();
            cellNext.clear();
            cellHead.resize(NumCells);
            Numparticles = _particles.size();
            cellNext.resize(Numparticles);
            this->reset_linkedlist();
            std::cout << "Numpoints=" << Numparticles << "\n";
            std::cout << "Ncells=" << Ncells.x << " " << Ncells.y << " " << Ncells.z << "\n";
            std::cout << "cellHead.size()=" << cellHead.size() << "\n";
            std::cout << "cellNext.size()=" << cellNext.size() << "\n";
            std::cout << "cellneighborlist.size()=" << cellneighborlist.size() << "\n";
        }
        else
        {
            std::cerr << "Error lcut must be smaller than the box " << std::endl;
            exit(0);
        }
    }
    void reset_linkedlist(void);
    void fill_linkedlist(void);
    std::map<std::string, std::vector<int>> get_linkedlist(void);

public:
    //linked list variables
    inth3 Ncells;                       ///< numbers of cell in each direction < x,y,z >
    int NumCells;                       ///< total number of cells
    real3 LengthCells;                  ///< length of the cell in each direction < x,y,z >
    host::vector<int> cellHead;         ///< which stores for each cell the first inserted element id;
    host::vector<int> cellNext;         ///< which stores for each entry the id of the next inserted element in the list.
    host::vector<int> cellneighborlist; ///< stores the neibourlist for the cells    device::vector<int> neighborlist;
protected:
    //particle variables
    int Numparticles;
    host::vector<ParticleType> &_particles;
    real lcut, lcut2;
    BoxType &_box;
};

void LinkedListType::reset_linkedlist(void)
{
    host::fill(cellHead.begin(), cellHead.end(), -1);
    host::fill(cellNext.begin(), cellNext.end(), -1);
}

void LinkedListType::fill_linkedlist(void)
{
    //reset the linked list
    this->reset_linkedlist();
    //loop over particles assign the cellID
    for (int pindex = 0; pindex < Numparticles; pindex ++)
    {
        int rx = ((int)((_particles[pindex].r.x + 0.5 * _box.L.x) / LengthCells.x));
        int ry = ((int)((_particles[pindex].r.y + 0.5 * _box.L.y) / LengthCells.y));
        int rz = ((int)((_particles[pindex].r.z + 0.5 * _box.L.z) / LengthCells.z));
        int cell_index = rx + Ncells.x * (ry + Ncells.y * rz);
        //exchange the last particle address
        int lastStartElement = cellHead[cell_index];
        cellHead[cell_index] = pindex;
        cellNext[pindex] = lastStartElement;
        //update particles
        _particles[pindex].cellId = cell_index;
    }
}

std::map<std::string, std::vector<int>> LinkedListType::get_linkedlist(void)
{
    std::map<std::string, std::vector<int>> cellHeadNext_map;
    cellHeadNext_map["Head"] = host::copy(cellHead);
    cellHeadNext_map["Next"] = host::copy(cellNext);
    return cellHeadNext_map;
}

#endif
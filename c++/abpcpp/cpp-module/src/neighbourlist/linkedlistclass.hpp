#ifndef __linkedlistclass_HPP__
#define __linkedlistclass_HPP__

#include <iostream>
#include <map>
#include "../types/globaltypes.hpp"
#include "../system/systemclass.hpp"
#include "../box/box.hpp"

class LinkedListType
{
    public:
    LinkedListType(SystemClass &system) : _system(system)
    {
    }
    ~LinkedListType() {}
    //linkedlist functions
    void create_linkedlist(real _lcut)
    {
        //retrieve the box
        auto box = _system.get_box();
        //retrieve the box
        lcut = _lcut;
        lcut2 = lcut * lcut;
        Ncells.x = ((int)(box.L.x / (lcut)));      ///< Numbers of cells x
        Ncells.y = ((int)(box.L.y / (lcut)));      ///< Numbers of cells y
        Ncells.z = ((int)(box.L.z / (lcut)));      ///< Numbers of cells z
        LengthCells.x = (real)(box.L.x / Ncells.x); ///< Length of the cells x
        LengthCells.y = (real)(box.L.y / Ncells.y); ///< Length of the cells y
        LengthCells.z = (real)(box.L.z / Ncells.z); ///< Length of the cells z
        NumCells = Ncells.x * Ncells.y * Ncells.z;
        if (NumCells > 0)
        {
            cellHead.clear();
            cellNext.clear();
            cellHead.resize(NumCells);
            Numparticles = _system.particles.size();
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
    SystemClass &_system;
    real lcut, lcut2;

};



#endif
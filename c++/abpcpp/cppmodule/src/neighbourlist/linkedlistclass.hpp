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
#ifndef __linkedlistclass_hpp__
#define __linkedlistclass_hpp__

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
        LengthCells.x = (real)(box.L.x / Ncells.x); ///< Length of the cells x
        LengthCells.y = (real)(box.L.y / Ncells.y); ///< Length of the cells y
        NumCells = Ncells.x * Ncells.y;
        if (NumCells > 0)
        {
            cellHead.clear();
            cellNext.clear();
            cellHead.resize(NumCells);
            Numparticles = _system.particles.size();
            cellNext.resize(Numparticles);
            this->reset_linkedlist();
            std::cout << "Numpoints=" << Numparticles << "\n";
            std::cout << "Ncells=" << Ncells.x << " " << Ncells.y << "\n";
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
    inth2 Ncells;                       ///< numbers of cell in each direction < x,y >
    int NumCells;                       ///< total number of cells
    real2 LengthCells;                  ///< length of the cell in each direction < x,y >
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
#include "linkedlistclass.hpp"

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
    //retrieve the box from system
    auto box = _system.get_box();
    for (int pindex = 0; pindex < Numparticles; pindex ++)
    {
        inth2 cellId;
        cellId.x = ((int)((_system.particles[pindex].r.x + 0.5 * box.L.x) / LengthCells.x));
        cellId.y = ((int)((_system.particles[pindex].r.y + 0.5 * box.L.y) / LengthCells.y));
        int cell_index = cellId.x + Ncells.x * cellId.y;
        //exchange the last particle address
        int lastStartElement = cellHead[cell_index];
        cellHead[cell_index] = pindex;
        cellNext[pindex] = lastStartElement;
        //update particles
        _system.particles[pindex].cellId = cellId;
    }
}

std::map<std::string, std::vector<int>> LinkedListType::get_linkedlist(void)
{
    std::map<std::string, std::vector<int>> cellHeadNext_map;
    cellHeadNext_map["Head"] = host::copy(cellHead);
    cellHeadNext_map["Next"] = host::copy(cellNext);
    return cellHeadNext_map;
}
#include "linkedlistclass.hpp"

void LinkedListType::reset_linkedlist(void)
{
    device::fill(cellHead.begin(), cellHead.end(), -1);
    device::fill(cellNext.begin(), cellNext.end(), -1);
}

DEV_LAUNCHABLE
void fill_linkedlist_kernel(const int Numparticles,
                            ParticleType *particles,
                            const BoxType box,
                            const real2 LengthCells,
                            const inth2 Ncells,
                            int *cellHead,
                            int *cellNext)
{
    //loop over particles assign the cellID
    for (int pindex = blockIdx.x * blockDim.x + threadIdx.x;
         pindex < Numparticles;
         pindex += blockDim.x * gridDim.x)
    {
        inth2 cellId;
        cellId.x = ((int)((particles[pindex].r.x + 0.5 * box.L.x) / LengthCells.x));
        cellId.y = ((int)((particles[pindex].r.y + 0.5 * box.L.y) / LengthCells.y));
        int cell_index = cellId.x + Ncells.x * cellId.y;
        //exchange the last particle address
        int lastStartElement = atomicExch(&cellHead[cell_index], pindex);
        cellNext[pindex] = lastStartElement;
        //update particles
        particles[pindex].cellId = cellId;
    }
}

void LinkedListType::fill_linkedlist(void)
{
    //reset the linked list
    this->reset_linkedlist();
    //retrieve the box from system
    auto box = _system.get_box();
    fill_linkedlist_kernel<<<_system._ep.getGridSize(), _system._ep.getBlockSize()>>>(_system.Numparticles,
                                                                                      device::raw_pointer_cast(&_system.particles[0]),
                                                                                      _system.get_box(),
                                                                                      LengthCells,
                                                                                      Ncells,
                                                                                      device::raw_pointer_cast(&cellHead[0]),
                                                                                      device::raw_pointer_cast(&cellNext[0]));
}
std::map<std::string, std::vector<int>> LinkedListType::get_linkedlist(void)
{
    std::map<std::string, std::vector<int>> cellHeadNext_map;
    cellHeadNext_map["Head"] = device::copy(cellHead);
    cellHeadNext_map["Next"] = device::copy(cellNext);
    return cellHeadNext_map;
}
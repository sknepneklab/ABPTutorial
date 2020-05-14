#include "neighbourlistclass.hpp"

void NeighbourListType::fill_neighbourlist(void)
{
    //fill the linked list
    this->fill_linkedlist();
    //loop over all the particles and build the neighbourlist
    //retrieve the box from system
    auto box = _system.get_box();
    for (int pindex_i = 0; pindex_i < Numparticles; pindex_i++)
    {
        auto cellId = _system.particles[pindex_i].cellId;
        int cell_index = cellId.x + Ncells.x * cellId.y ;
        _system.particles[pindex_i].coordination = 0; //no neighbours
        old_positions[pindex_i] = _system.particles[pindex_i].r;
        for (int nx = -1; nx <= 1; nx++)
            for (int ny = -1; ny <= 1; ny++)
            {
                auto cellId_j = cellId;
                cellId_j.x += nx;
                cellId_j.y += ny;

                if (box.periodic.x)
                {
                    ///< X
                    if (cellId_j.x < 0)
                        cellId_j.x = Ncells.x - 1;
                    else if (cellId_j.x > (Ncells.x - 1))
                        cellId_j.x = 0;
                }
                if (box.periodic.y)
                {
                    ///< Y
                    if (cellId_j.y < 0)
                        cellId_j.y = Ncells.y - 1;
                    else if (cellId_j.y > (Ncells.y - 1))
                        cellId_j.y = 0;
                }
                bool flag = (cellId_j.y > (Ncells.y - 1)) || (cellId_j.x > (Ncells.x - 1)) || (cellId_j.y < 0) || (cellId_j.x < 0);
                if (!flag)
                {
                    int cell_index_j = cellId.x + Ncells.x * cellId.y;
                    //Now loop over the cell_index_j and add it to the neighbours list of i
                    int pindex_j = cellHead[cell_index_j];
                    while (pindex_j != -1)
                    {
                        if (pindex_i != pindex_j)
                        {
                            real2 rij = host::minimum_image(_system.particles[pindex_i].r, _system.particles[pindex_j].r, box);
                            real rij2 = vdot(rij, rij);
                            if (rij2 < rcut2)
                            {
                                int ng_index = _system.particles[pindex_i].coordination + max_ng_per_particle * pindex_i;
                                neighbourlist[ng_index] = pindex_j;
                                _system.particles[pindex_i].coordination++;
                            }
                        }
                        pindex_j = cellNext[pindex_j];
                    }
                }
            }
    }
}

void NeighbourListType::automatic_update(void)
{
    //loop over all the particles and check if the neibourlist need to be updated
    bool need_update = false;
    //retrieve the box from system
    auto box = _system.get_box();
    for (int pindex_i = 0; pindex_i < _system.Numparticles; pindex_i++)
    {
        ParticleType pi = _system.particles[pindex_i];
        real2 rij = host::minimum_image(_system.particles[pindex_i].r, old_positions[pindex_i], box);
        real rij2 = vdot(rij, rij);
        if (rij2 >= 0.25 * skin2)
        {
            need_update = true;
            break;
        }
    }
    if (need_update)
        this->fill_neighbourlist();
}
host::vector<int> NeighbourListType::get_neighbourlist(void)
{
    host::vector<int> neig;
    return neig;
}
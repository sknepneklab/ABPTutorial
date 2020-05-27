#include "neighbourlistclass.hpp"
#include "../box/pbc_device.hpp"
#include "../configuration/atomic_exch_double.hpp"

DEV_LAUNCHABLE
void fill_neighbourlist_kernel(const int Numparticles,
                               ParticleType *particles,
                               real2 *old_positions,
                               const BoxType box,
                               const inth2 Ncells,
                               const int *__restrict__ cellHead,
                               const int *__restrict__ cellNext,
                               const int max_ng_per_particle,
                               const real rcut2,
                               int *nglist)
{
    //loop over all the particles and build the neighbourlist
    for (int pindex_i = blockIdx.x * blockDim.x + threadIdx.x;
         pindex_i < Numparticles;
         pindex_i += blockDim.x * gridDim.x)
    {
        auto cellId = particles[pindex_i].cellId;
        int coordination = 0; //no neighbours
        old_positions[pindex_i] = particles[pindex_i].r;
#pragma unroll(3)
        for (int nx = -1; nx <= 1; nx++)
        {
#pragma unroll(3)
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
                    int cell_index_j = cellId_j.x + Ncells.x * cellId_j.y;
                    //Now loop over the cell_index_j and add it to the neighbours list of i
                    int pindex_j = cellHead[cell_index_j];
                    while (pindex_j != -1)
                    {
                        if (pindex_i != pindex_j)
                        {
                            real2 rij = device::minimum_image(particles[pindex_i].r, particles[pindex_j].r, box);
                            real rij2 = vdot(rij, rij);
                            if (rij2 < rcut2)
                            {
                                int ng_index = coordination + max_ng_per_particle * pindex_i;
                                nglist[ng_index] = pindex_j;
                                coordination++;
                            }
                        }
                        pindex_j = cellNext[pindex_j];
                    }
                }
            }
        }
        particles[pindex_i].coordination = coordination;
    }
}

DEV_LAUNCHABLE
void fill_neighbourlist_brute_kernel(const int Numparticles,
                               ParticleType *particles,
                               real2 *old_positions,
                               const BoxType box,
                               const int max_ng_per_particle,
                               const real rcut2,
                               int *nglist)

{
    //loop over all the particles and build the neighbourlist
    for (int pindex_i = blockIdx.x * blockDim.x + threadIdx.x;
         pindex_i < Numparticles;
         pindex_i += blockDim.x * gridDim.x)
    {
        int coordination = 0; //no neighbours
        old_positions[pindex_i] = particles[pindex_i].r;
        for(int pindex_j = 0; pindex_j<Numparticles; pindex_j++)
        {
            if (pindex_i != pindex_j)
            {
                real2 rij = device::minimum_image(particles[pindex_i].r, particles[pindex_j].r, box);
                real rij2 = vdot(rij, rij);
                if (rij2 < rcut2)
                {
                    int ng_index = coordination + max_ng_per_particle * pindex_i;
                    nglist[ng_index] = pindex_j;
                    coordination++;
                }
            }
        }
        particles[pindex_i].coordination = coordination;
    }

}                               


void NeighbourListType::fill_neighbourlist(void)
{

    //fill the linked list
    this->fill_linkedlist();
    //retrieve the box from system
    auto box = _system.get_box();
   fill_neighbourlist_kernel<<<_system._ep.getGridSize(), _system._ep.getBlockSize()>>>(_system.Numparticles,
                                                                                         device::raw_pointer_cast(&_system.particles[0]),
                                                                                         device::raw_pointer_cast(&old_positions[0]),
                                                                                         _system.get_box(),
                                                                                         Ncells,
                                                                                         device::raw_pointer_cast(&cellHead[0]),
                                                                                         device::raw_pointer_cast(&cellNext[0]),
                                                                                         max_ng_per_particle,
                                                                                         rcut2,
                                                                                         device::raw_pointer_cast(&nglist[0]));
    
     /*fill_neighbourlist_brute_kernel<<<_system._ep.getGridSize(), _system._ep.getBlockSize()>>>(_system.Numparticles,
                                                                                         device::raw_pointer_cast(&_system.particles[0]),
                                                                                         device::raw_pointer_cast(&old_positions[0]),
                                                                                         _system.get_box(),
                                                                                         max_ng_per_particle,
                                                                                         rcut2,
                                                                                         device::raw_pointer_cast(&nglist[0]));*/
    
    real2 value;
    value.x = value.y = 0.0;
    old_positions[_system.Numparticles] = value;
}

void NeighbourListType::fill_neighbourlist_brute_force(void)
{
    this->fill_neighbourlist();
}

DEV_LAUNCHABLE
void automatic_update_kernel(const int Numparticles,
                             const ParticleType *__restrict__ particles,
                             real2 *old_positions,
                             const BoxType box,
                             const real skin2)
{
    //loop over all the particles and build the neighbourlist
    for (int pindex_i = blockIdx.x * blockDim.x + threadIdx.x;
         pindex_i < Numparticles;
         pindex_i += blockDim.x * gridDim.x)
    {
        real2 rij = device::minimum_image(particles[pindex_i].r, old_positions[pindex_i], box);
        real rij2 = vdot(rij, rij);
        if (rij2 >= 0.25 * skin2)
        {
            real lastStartElement = device::double_atomicExch(&old_positions[Numparticles].x, -1.0);
        }
    }
}

void NeighbourListType::automatic_update(void)
{
    //this->fill_neighbourlist();

    automatic_update_kernel<<<_system._ep.getGridSize(), _system._ep.getBlockSize()>>>(_system.Numparticles,
                                                                                       device::raw_pointer_cast(&_system.particles[0]),
                                                                                       device::raw_pointer_cast(&old_positions[0]),
                                                                                       _system.get_box(),
                                                                                       skin2);


    real2 value = old_positions[_system.Numparticles];
    if (value.x < 0)
    {
        //std::cout<< "NeighbourList auto update hit"<< std::endl;
        this->fill_neighbourlist();
    }
    
}

std::map<std::string, host::vector<int>> NeighbourListType::get_neighbourlist(void)
{
    std::map<std::string, std::vector<int>> cellHeadNext_map;
    cellHeadNext_map["Head"] = device::copy(cellHead);
    cellHeadNext_map["Next"] = device::copy(cellNext);
    cellHeadNext_map["List"] = device::copy(nglist);
    return cellHeadNext_map;
}
#ifndef __BOX_HPP__
#define __BOX_HPP__

#include "../types/globaltypes.hpp"

struct BoxType 
{
    real3 L;             //!< box length Lhi-Llo
    real3 Llo;           //!< low value of the of the box
    real3 Lhi;           //!< high value of the of the box
    bool3 periodic;      //!< periodicity of the box
};
#endif 
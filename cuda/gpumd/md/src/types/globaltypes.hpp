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
#ifndef __globaltypes_hpp__
#define __globaltypes_hpp__

#include <iostream>
#include <sstream>
#include <math.h>

/** @} */            
#define BIG_ENERGY_LIMIT  1e15       //!< Effectively acts as infinity
#define defPI 3.141592653589793
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

/** @brief real type **/
using real = double;    

/** @brief 2D xy type **/
template <typename T>
struct xyType{T x,y;};
using real2 = xyType<real>;
using inth2 = xyType<int>;
using bool2 = xyType<bool>;


/** @brief Vector addition which will be use in the gpu easily */
#define vsum(v,v1,v2)               \
            (v.x = v1.x + v2.x),     \
            (v.y = v1.y + v2.y)     \

/** @brief Vector subtraction. */
#define vsub(v,v1,v2)               \
            (v.x = v1.x - v2.x),     \
            (v.y = v1.y - v2.y)     \

/** @brief Vector dot product. */
#define vdot(v1,v2)   (v1.x*v2.x  + v1.y*v2.y)

/** @brief Vector cross product. */
#define vcross(v1,v2) (v1.x*v2.y  -  v1.y*v2.x)

/** @brief Constant times a vector **/
#define aXvec(a,v)                       \
            (v.x = (a)*v.x),             \
            (v.y = (a)*v.y)             \
           
/**
 * @brief to_string_scientific converts any number to scientific notation
 * @tparam T 
 * @param a_value value to be converted to
 * @return std::string string scientific notation formated
 */
template <typename T>
std::string to_string(const T a_value)
{
    std::ostringstream out;
    out << a_value;
    return out.str();
}
#endif

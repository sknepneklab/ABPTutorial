#ifndef __GLOBALTYPES_HPP__
#define __GLOBALTYPES_HPP__

#include <vector>

/** @} */            
#define BIG_ENERGY_LIMIT  1e15       //!< Effectively acts as infinity
#define defPI 3.141592653589793
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

/** @brief real type **/
using real = double;    

/** @brief 3D xyz type **/
template <typename T>
struct xyzT{T x,y,z;};
using real3 = xyzT<real>;
using inth3 = xyzT<int>;
using bool3 = xyzT<bool>;


/** @brief Vector addition which will be use in the gpu easily */
#define vsum(v,v1,v2)               \
            (v.x = v1.x + v2.x),     \
            (v.y = v1.y + v2.y),     \
            (v.z = v1.z + v2.z)     \

/** @brief Vector subtraction. */
#define vsub(v,v1,v2)               \
            (v.x = v1.x - v2.x),     \
            (v.y = v1.y - v2.y),     \
            (v.z = v1.z - v2.z)     \

/** @brief Vector dot product. */
#define vdot(v1,v2)   (v1.x*v2.x  + v1.y*v2.y  + v1.z*v2.z)

/** @brief Vector cross product. */
#define vcross(v,v1,v2)                         \
            (v.x = v1.y*v2.z  -  v1.z*v2.y),     \
            (v.y = v1.z*v2.x  -  v1.x*v2.z),     \
            (v.z = v1.x*v2.y  -  v1.y*v2.x)     \

/** brief Constant times a vector **/
#define aXvec(a,v)                       \
            (v.x = (a)*v.x),             \
            (v.y = (a)*v.y),             \
            (v.z = (a)*v.z)              \
            

namespace host
{
    using namespace std;
    template <typename T>
    using vector = std::vector<T>;
} // namespace host

/**
 * @brief to_string_scientific converts any number to scientific notation
 * 
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
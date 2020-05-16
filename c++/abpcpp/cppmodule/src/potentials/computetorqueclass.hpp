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
#ifndef __computetorqueclass_hpp__
#define __computetorqueclass_hpp__

/** @defgroup computeenergy Compute Vertex Energy
 *  @brief ComputeTorqueClass abstracts definitions
 *  @{
 */
#include <memory>
#include <iostream>

#include "../types/globaltypes.hpp"
#include "../system/systemclass.hpp"
/**
 * @class ComputeTorqueClass
 * @brief ComputeTorqueClass abstract class for compute different potentials, forces and torques
 */

class ComputeTorqueClass
{
public:
  /**
   * @brief ComputeTorqueClass constructor
   * @param SystemClass reference to the system
   */
  ComputeTorqueClass(SystemClass &system) : _system(system)
  {
    rcut = 0.0;
  }
  /**
     * @brief ComputeTorqueClass Destructor
     */
  virtual ~ComputeTorqueClass() {}
  /**
     * @brief compute energy for the actual configuration
     * @param void
     * @return void 
     */
  virtual void compute_energy(void){};
  /**
     * @brief compute force for the actual configuration
     * @param void
     * @return void 
     */
  virtual void compute(void){};
  /**
     * @brief Get the name object
     * @return std::string 
     */
  std::string get_name(void) { return name; }
  /**
     * @brief Get the type object
     * @return std::string 
     */
  std::string get_type(void) { return type; }
  /**
     * @brief Get the pair potential's cut off radius 
     * @return std::string 
   */
  real get_rcut(void) { return rcut; }
  /**
     * @brief Set property
    */
  virtual void set_default_properties(void) = 0;
  virtual void set_property(const std::string &prop_name, const double &value) { this->print_warning_calling("double "); };
  void print_warning_calling(const std::string &message) { std::cerr << "torque " << name << " cannot be called with " << message << "\n"; }
  void print_warning_property_name(const std::string &message) { std::cerr << "parameter " << message << " is not part of " << name << " potential \n"; }

protected:
  SystemClass &_system; //!< Reference to the system
  std::string name;     //!< Name declared for that potential
  std::string type;     //!< Potential type, active, torque, conservative, etc
  real rcut;            //!< maximum cut off radius for a given potential
};

typedef std::unique_ptr<ComputeTorqueClass> ComputeTorqueClass_ptr;

#endif

/** @} */

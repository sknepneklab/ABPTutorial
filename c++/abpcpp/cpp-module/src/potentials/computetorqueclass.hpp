#ifndef __computetorqueclass_HPP__
#define __computetorqueclass_HPP__

/** @defgroup computeenergy Compute Vertex Energy
 *  @brief ComputeTorqueClass abstracts definitions
 *  @{
 */
#include <memory>
#include <map>
#include <iostream>

#include "../types/globaltypes.hpp"
#include "../particle/particles.hpp"
/**
 * @class ComputeTorqueClass
 * @brief ComputeTorqueClass abstract class for compute different potentials, forces and torques
 */

class ComputeTorqueClass
{
public:
   /**
   * @brief ComputeTorqueClass constructor
   * @param ParticlesClass reference to the particles
   */
   ComputeTorqueClass(ParticlesClass &particles) : _particles(particles)
   {
      NUM_TYPES_PAIR = (int)sqrt(1.0 * 100);
      NUM_TYPES_ALLOWED = NUM_TYPES_PAIR * NUM_TYPES_PAIR;
      max_rcut = 0.0;
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
   real get_rcut(void) { return max_rcut; }
   /**
     * @brief Set property
    */
   virtual void set_defaults_property(void) = 0;
   virtual void set_property(std::string, std::map<int, double> &) { this->print_warning_calling("int, double "); };
   virtual void set_property(std::string, std::map<int, int> &) { this->print_warning_calling("int, int "); }
   virtual void set_property(std::string, std::map<int, bool> &) { this->print_warning_calling("int, bool "); }
   virtual void set_property(std::string, std::map<int, std::string> &) { this->print_warning_calling("int,string"); }
   virtual void set_property(std::string, std::map<int, std::vector<double>> &) { this->print_warning_calling("int, vector<double>"); }
   virtual void set_property(std::string, std::map<int, std::vector<real3>> &) { this->print_warning_calling("int, vector<real3>"); }
   virtual void set_property(std::string, std::map<std::pair<int, int>, double> &) { this->print_warning_calling("pair<int, int>, double"); };
   void print_warning_calling(std::string message) { std::cerr << "potential " << name << " cannot be called with " << message << "\n"; }
   void print_warning_property_name(std::string message) { std::cerr << "parameter " << message << " is not part of " << name << " potential \n"; }

protected:
   ParticlesClass &_particles; //!< Reference to the particles
   std::string name;           //!< Name declared for that potential
   std::string type;           //!< Potential type, active, torque, conservative, etc
   int NUM_TYPES_ALLOWED;      //!< Number of type allowed
   int NUM_TYPES_PAIR;         //!<
   real max_rcut;              //!< maximum cut off radius for a given potential
};

typedef std::shared_ptr<ComputeTorqueClass> ComputeTorqueClass_ptr;

#endif

/** @} */

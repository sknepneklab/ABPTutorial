#ifndef __integratorclass__HPP__
#define __integratorclass__HPP__

#include <memory>
#include <iostream>

#include "../types/globaltypes.hpp"
#include "../system/systemclass.hpp"

/**
 * @class IntegratorClass
 * @brief Abstract class for particle integrator
 */
class IntegratorClass
{
public:
  /**
   * @brief IntegratorClass Constructor
   * 
   * @param potentials pointer to the loaded potentials
   * @param param pointer to the integrator parameters
   */
  IntegratorClass(SystemClass &system) : _system(system)
  {
  }
  /**
     * @brief IntegratorClass Destructor
     */
  virtual ~IntegratorClass()
  {
  }
  /**
     * @brief abstract pre step integrator function
     * @param void 
     */
  virtual void prestep(void) = 0;
  /**
     * @brief abstract pre step integrator function
     * @param void 
     */
  virtual void poststep(void) = 0;
  /**
     * @brief set the integration temperature
     * @param temperature
     */
  void set_temperature(real _T)
  {
    T = fabs(_T);
    update_temperature_parameters();
  }
  /**
   * @brief update the parameters that depend on the temperature
   * @return void 
   */
  virtual void update_temperature_parameters() {}

  /**
     * @brief Get the temperature for the integrator object
     * @return real 
     */
  real get_temperature(void) const { return T; }
  /**
     * @brief set the integration temperature
     * @param temperature
     */
  void set_time_step(real _dt)
  {
    dt = fabs(_dt);
    update_time_step_parameters();
  }
  /**
   * @brief update the parameters that depend on the time step
   * @return void 
   */
  virtual void update_time_step_parameters() {}
  /**
     * @brief Get the time step for the integrator object
     * @return real 
     */
  real get_time_step(void) const { return dt; }
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

  virtual void set_defaults_property(void) = 0;
  virtual void set_property(const std::string&, const double &) { this->print_warning_calling("double"); }
  void print_warning_calling(std::string message) { std::cerr << "integrator " << name << "cannot be called with " << message << "\n"; }
  void print_warning_property_name(std::string message) { std::cerr << "parameter " << message << " is not part of " << name << " integrator \n"; }

protected:
  SystemClass &_system;      //!< reference to system class where the box and particles are stored
  std::string name;          //!< Integrator name
  std::string type;          //!< integrator type, active, torque, etc

private:
  real T;  //!< Temperature of the system
  real dt; //!< Time Step
};

typedef std::shared_ptr<IntegratorClass> IntegratorClass_ptr;


#endif

/** @} */

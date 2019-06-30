/* -------------------------------------------------------------------------
 * iblevelsetfunctions.h
 *
 * This classes provides a mother class for ib level set functions
 * an ib level set function must provide it's own way to calculate
 * the distance and possibly, the velocity and the value of a scalar
 * ------------------------------------------------------------------------
 *
 * Author: Bruno Blais, Polytechnique Montreal, 2019-
 */


#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>


using namespace dealii;


#ifndef LETHE_IBLEVELSETFUNCTIONS_H
#define LETHE_IBLEVELSETFUNCTIONS_H

template <int dim>
class IBLevelSetFunctions
{
public:
    IBLevelSetFunctions();
    IBLevelSetFunctions(double p_scal=0):
      scalar_value(p_scal)
    {}

    // Value of the distance
    virtual double distance(const Point<dim> &p) = 0;
    virtual double scalar  (const Point<dim> &/*p*/) {return scalar_value;}
    virtual void   velocity(const Point<dim> &p, Vector<double> &values)=0;
protected:
      double   scalar_value; // will be used to make tests by solving the heat equation
};


template <int dim>
class IBLevelSetCircle: public IBLevelSetFunctions<dim>
{
private:
  Point<dim>    center;
  Tensor<1,dim> linear_velocity;
  Tensor<1,3>   angular_velocity; // rad/s

  double radius;
  bool inside;


public:
  IBLevelSetCircle(Point<dim> p_center, Tensor<1,dim> p_linear_velocity, Tensor<1,3> p_angular_velocity, double p_scal, bool p_fluid_inside, double p_radius):
    IBLevelSetFunctions<dim>(p_scal),
    center(p_center),
    linear_velocity(p_linear_velocity),
    angular_velocity(p_angular_velocity),
    radius(p_radius),
    inside(p_fluid_inside) {}

    // Value of the distance
    virtual double distance(const Point<dim> &p)
    {
      const double x = p[0]-this->center[0];
      const double y = p[1]-this->center[1];;

      if (!inside){return std::sqrt(x*x+y*y)-radius;}
      else {return -(std::sqrt(x*x+y*y)-radius);}

    }
    virtual void   velocity(const Point<dim> &p, Vector<double> &values)
    {
      if (dim==2)
      {
        const double x = p[0]-this->center[0];
        const double y = p[1]-this->center[1];
        const double omega = this->angular_velocity[2];
        values[0] = -omega*y+this->linear_velocity[0];
        values[1] = omega*x+this->linear_velocity[1];
      }
    }

protected:
};


// Level Set function for a plane
// center is a point lying on the plane
// normal is the outward normal of the plane

template <int dim>
class IBLevelSetPlane: public IBLevelSetFunctions<dim>
{
private:
  Point<dim>    center;
  Tensor<1,dim> normal;
  Tensor<1,dim> linear_velocity;
public:
    IBLevelSetPlane(Point<dim> p_center, Tensor<1,dim> p_normal, Tensor<1,dim> p_linear_velocity, double p_scal):
      IBLevelSetFunctions<dim>(p_scal),
      center(p_center),
      normal(p_normal),
      linear_velocity(p_linear_velocity)
    {
      // Normalise normal vector
      normal = normal / normal.norm();
    }

    // Value of the distance
    virtual double distance(const Point<dim> &p)
    {
      return ((p-center) * normal);
    }

    // Value of the velocity
    virtual void velocity(const Point<dim> &/*p*/, Vector<double> &values)
    {
        for (int i = 0 ; i<dim ; ++i)
        {
          values[i] = this->linear_velocity[i];
        }
    }
};

#endif

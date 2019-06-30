/* -------------------------------------------------------------------------
 * ibcombiner.h
 *
 * This classes combines the ib level set functions in to a single function
 * This function can be used to query the distance, velocity and scalar
 * values of points or list of points
 * For the distance, it inherits the value function from it's Function
 * base class
 * ------------------------------------------------------------------------
 *
 * Author: Bruno Blais, Polytechnique Montreal, 2019-
 */


#include <deal.II/base/function.h>
#include "iblevelsetfunctions.h"

#include <deal.II/base/timer.h>


using namespace dealii;


#ifndef LETHE_IBCOMBINER_H
#define LETHE_IBCOMBINER_H

template <int dim>
class IBCombiner: public Function<dim>
{
public:
  IBCombiner(std::vector<IBLevelSetFunctions<dim>* > p_functions) :
    Function<dim>(1),
    functions(p_functions)
  {}

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &return_values) const;

  virtual double value(const Point<dim> &p) const;

  double scalar(const Point<dim> &p) const;

  void velocity(const Point<dim> &p, Vector<double> &velocity_values) const;

  void setFunctions(std::vector<IBLevelSetFunctions<dim>* > p_functions)
  {
    functions=p_functions;
  }

  inline unsigned int localize(const Point<dim> &p) const
  {
    double dist=DBL_MAX;
    unsigned int shape_id=-1;
    for (unsigned ib=0 ; ib < functions.size() ; ++ib)
    {
     double local_dist=functions[ib]->distance(p);
     if (local_dist<dist)
       shape_id = ib;
     }
    return shape_id;
  }

private:
  std::vector<IBLevelSetFunctions<dim>* >     functions;

};

template<int dim>
double IBCombiner<dim>::value(const Point<dim> &p) const
{
  double dist=DBL_MAX;
  for (unsigned ib=0 ; ib < functions.size() ; ++ib)
    dist = std::min(dist,functions[ib]->distance(p));

  return dist;
}

template<int dim>
void IBCombiner<dim>::value_list(const std::vector<Point<dim>> &points,
                                 std::vector<double>           &return_values) const
{
  assert(points.size()==return_values.size());
  for (unsigned int ipt =0 ; ipt < points.size() ; ++ipt)
  {
    double dist=DBL_MAX;
    for (unsigned ib=0 ; ib < functions.size() ; ++ib)
      dist = std::min(dist,functions[ib]->distance(points[ipt]));

   return_values[ipt]=dist;
  }
}

template<int dim>
double IBCombiner<dim>::scalar(const Point<dim> &p) const
{
  unsigned int shape_id = localize(p);
  return functions[shape_id]->scalar(p);
}

template<int dim>
void IBCombiner<dim>::velocity(const Point<dim> &p, Vector<double> &velocity_values) const
{
  unsigned int shape_id = localize(p);
  return functions[shape_id]->velocity(p,velocity_values);
}


#endif // IBCOMPOSER_H

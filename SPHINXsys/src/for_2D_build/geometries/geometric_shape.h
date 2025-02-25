/* -------------------------------------------------------------------------*
 *								SPHinXsys									*
 * --------------------------------------------------------------------------*
 * SPHinXsys (pronunciation: s'finksis) is an acronym from Smoothed Particle	*
 * Hydrodynamics for industrial compleX systems. It provides C++ APIs for	*
 * physical accurate simulation and aims to model coupled industrial dynamic *
 * systems including fluid, solid, multi-body dynamics and beyond with SPH	*
 * (smoothed particle hydrodynamics), a meshless computational method using	*
 * particle discretization.													*
 *																			*
 * SPHinXsys is partially funded by German Research Foundation				*
 * (Deutsche Forschungsgemeinschaft) DFG HU1527/6-1, HU1527/10-1				*
 * and HU1527/12-1.															*
 *                                                                           *
 * Portions copyright (c) 2017-2020 Technical University of Munich and		*
 * the authors' affiliations.												*
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may   *
 * not use this file except in compliance with the License. You may obtain a *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.        *
 *                                                                           *
 * --------------------------------------------------------------------------*/
/**
 * @file geometric_shape.h
 * @brief Here, we define shapes represented directly by geometric elements.
 * @details The simbody contact geometry is used.
 * @author	Xiangyu Hu
 */

#ifndef GEOMETRIC_SHAPE_H
#define GEOMETRIC_SHAPE_H

#include "base_geometry.h"
#include "multi_polygon_shape.h"

namespace SPH
{
    class GeometricShapeBox : public Shape
    {
    public:
        explicit GeometricShapeBox(const Vec2d &halfsize,
                                   const std::string &shape_name = "GeometricShapeBox");
        virtual ~GeometricShapeBox(){};

        virtual bool checkContain(const Vec2d &pnt, bool BOUNDARY_INCLUDED = true) override;
        virtual Vec2d findClosestPoint(const Vec2d &pnt) override;

    protected:
        Vec2d halfsize_;
   		MultiPolygon multi_polygon_;


        virtual BoundingBox findBounds() override;
    };

    class GeometricShapeBall : public Shape
    {
        Vec2d center_;
        Real radius_;

    public:
        explicit GeometricShapeBall(const Vec2d &center, Real radius,
                                    const std::string &shape_name = "GeometricShapeBall");
        virtual ~GeometricShapeBall(){};

        virtual bool checkContain(const Vec2d &pnt, bool BOUNDARY_INCLUDED = true) override;
        virtual Vec2d findClosestPoint(const Vec2d &pnt) override;
 
    protected:
        virtual BoundingBox findBounds() override;
    };
}

#endif // GEOMETRIC_SHAPE_H

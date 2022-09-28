/**
 * ...
 */

#ifndef VECTORIZATION_HELPER_H
#define VECTORIZATION_HELPER_H


#include "base_data_type.h"
#include "data_type.h"
#include <large_data_containers.h>
#include "xsimd/xsimd.hpp"


// TODO: put NTraits specialization in some other file, e.g. batch_sph.h
#include "SimTKcommon/internal/NTraits.h"
namespace SPH
{
	using RealBatchSph = xsimd::batch<Real>;

	// Two dimensional SimTK::Vec where each element is of
	// type xsimd::batch<Real>
	using VecdBatchSph = SimTK::Vec<2, RealBatchSph>;
}

template <>
class SimTK::NTraits<SPH::RealBatchSph>
{
public:                                         
    typedef SPH::RealBatchSph    T;
    typedef negator<T>       TNeg;              
    typedef T                TWithoutNegator;   
    typedef T                TReal;             
    typedef T                TImag;             
    typedef complex<T>       TComplex;          
    typedef T                THerm;             
    typedef T                TPosTrans;         
    typedef T                TSqHermT;          
    typedef T                TSqTHerm;          
    typedef T                TElement;          
    typedef T                TRow;              
    typedef T                TCol;              
    typedef T                TSqrt;             
    typedef T                TAbs;              
    typedef T                TStandard;         
    typedef T                TInvert;           
    typedef T                TNormalize;        
    typedef T                Scalar;            
    typedef T                ULessScalar;       
    typedef T                Number;            
    typedef T                StdNumber;         
    typedef T                Precision;         
    typedef T                ScalarNormSq;      

	template <class P>
	struct Result
	{
        typedef typename CNT<P>::template Result<T>::Mul Mul;
        typedef typename CNT< typename CNT<P>::THerm >::template Result<T>::Mul Dvd;
        typedef typename CNT<P>::template Result<T>::Add Add;
        typedef typename CNT< typename CNT<P>::TNeg >::template Result<T>::Add Sub;
    };                                          

    enum
    {
        NRows = 1, 
        NCols = 1, 
        RowSpacing = 1, 
        ColSpacing = 1, 
        NPackedElements = 1, 
        NActualElements = 1, 
        NActualScalars = 1, 
        ImagOffset = 0, 
        RealStrideFactor = 1, 
        ArgDepth = SCALAR_DEPTH, 
        IsScalar = 1, 
        IsULessScalar = 1, 
        IsNumber = 1, 
        IsStdNumber = 1, 
        IsPrecision = 1, 
        SignInterpretation = 1                 
    };
};

template <>
struct SimTK::Widest<SPH::RealBatchSph, SPH::RealBatchSph>
{
	typedef SPH::RealBatchSph Type;
	typedef SPH::RealBatchSph Precision;
};

template<>
struct SimTK::NTraits<SPH::RealBatchSph>::Result<SPH::RealBatchSph>
{
	typedef Widest<SPH::RealBatchSph, SPH::RealBatchSph>::Type Mul;
	typedef Mul Dvd;
	typedef Mul Add;
	typedef Mul Sub;
};

template <>
class SimTK::CNT<SPH::RealBatchSph> : public SimTK::NTraits<SPH::RealBatchSph> {};

namespace SPH
{
	// Size of a SIMD register of SPH::Real values for default ISA (in bytes)
	static const auto SIMD_REGISTER_SIZE_REAL_BYTES = sizeof(xsimd::types::simd_register<Real, xsimd::default_arch>);

	// Size of a SIMD register of SPH::Real values for default ISA (in elements)
	static const auto SIMD_REGISTER_SIZE_REAL_ELEMENTS = xsimd::simd_type<Real>::size;


	inline void InitWithDefaultValue(const Real default_value, xsimd::batch<Real>& reg_0)
	{
		alignas(SIMD_REGISTER_SIZE_REAL_BYTES) Real vec_0[SIMD_REGISTER_SIZE_REAL_ELEMENTS] = { default_value };
		reg_0 = xsimd::load_aligned(&vec_0[0]);
	}

	inline void InitWithDefaultValueVecdBatch(const Real default_value, VecdBatchSph& vec)
	{
		alignas(SIMD_REGISTER_SIZE_REAL_BYTES) Real vec_0[SIMD_REGISTER_SIZE_REAL_ELEMENTS] = { default_value };
		vec = VecdBatchSph(xsimd::load_aligned(&vec_0[0]), xsimd::load_aligned(&vec_0[0]));
	}

	// Alternative to xsimd::batch<T,A>::gather() for gathering a RealBatchSph
	inline RealBatchSph LoadReal(const size_t* idx, const StdLargeVec<Real>& indirect_indexed_data)
	{
		return
		{
			indirect_indexed_data[*idx],
			indirect_indexed_data[*(idx + 1)],
			indirect_indexed_data[*(idx + 2)],
			indirect_indexed_data[*(idx + 3)]
		};
	}

	// Alternative to xsimd::batch<T,A>::gather() for gathering a VecdBatchSph 
	inline VecdBatchSph LoadVecd(const size_t* idx, const StdLargeVec<Vecd>& indirect_indexed_data)
	{
		return
		{
			{	// Batch X
				indirect_indexed_data[*idx].get(0),
				indirect_indexed_data[*(idx + 1)].get(0),
				indirect_indexed_data[*(idx + 2)].get(0),
				indirect_indexed_data[*(idx + 3)].get(0)
			},
			{	// Batch Y
				indirect_indexed_data[*idx].get(1),
				indirect_indexed_data[*(idx + 1)].get(1),
				indirect_indexed_data[*(idx + 2)].get(1),
				indirect_indexed_data[*(idx + 3)].get(1)
			}
		};
	}

	template <class T>
	void EstimateError(T& res_scalar, T& res_vector)
	{
		std::cout << " error = " << res_scalar - res_vector << std::endl;
	}

	void WriteTwoValuesToFile(const std::string& file_name, long long value1, long long value2, char delimiter);

}
#endif // VECTORIZATION_HELPER_H
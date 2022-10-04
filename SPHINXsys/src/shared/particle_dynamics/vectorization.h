/**
 * ...
 */

#ifndef VECTORIZATION_HELPER_H
#define VECTORIZATION_HELPER_H


#include "base_data_type.h"
#include "data_type.h"
#include "large_data_containers.h"
#include "SimTKcommon/internal/NTraits.h"
#include "xsimd/xsimd.hpp"


namespace SPH
{
	class SingleBatchSph : public xsimd::batch<Real>
	{
	public:
		SingleBatchSph()
			: batch()
		{}

		SingleBatchSph(Real val) noexcept
			: batch(val)
		{}

		template <class... Ts>
		SingleBatchSph(Real val0, Real val1, Ts... vals) noexcept
			: batch(val0, val1, static_cast<Real>(vals)...)
		{}

		SingleBatchSph(batch o)
			: batch(o)
		{}

		SingleBatchSph& operator=(const xsimd::batch<Real>& o)
		{
			batch::operator=(o);
			return *this;
		}
	};
	// Alternatively xsimd::batch can be directly used as an element of SimTK::Vec
	// using SingleBatchSph = xsimd::batch<Real>;

	// Two dimensional SimTK::Vec where each element is of type xsimd::batch<Real>
	using VecdBatchSph = SimTK::Vec<2, SingleBatchSph>;
}

template <>
class SimTK::NTraits<SPH::SingleBatchSph>
{
public:                                         
    typedef SPH::SingleBatchSph T;
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
struct SimTK::Widest<SPH::SingleBatchSph, SPH::SingleBatchSph>
{
	typedef SPH::SingleBatchSph Type;
	typedef SPH::SingleBatchSph Precision;
};

template<>
struct SimTK::NTraits<SPH::SingleBatchSph>::Result<SPH::SingleBatchSph>
{
	typedef Widest<SPH::SingleBatchSph, SPH::SingleBatchSph>::Type Mul;
	typedef Mul Dvd;
	typedef Mul Add;
	typedef Mul Sub;
};

template <>
class SimTK::CNT<SPH::SingleBatchSph> : public SimTK::NTraits<SPH::SingleBatchSph> {};

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


	// Generalized template function for loading indirect indexed data into a single batch
	// (alternative to xsimd::batch<T,A>::gather() functions)
	template< int /*number of elements in a batch*/, class ContainerType>
	SingleBatchSph LoadSingleBatchSph(const size_t* /*idx*/, const StdLargeVec<ContainerType>& /*indirect_indexed_data*/)
	{
		return{};
	}

	// Specialization for a batch packed with 4 values
	template<>
	inline SingleBatchSph LoadSingleBatchSph<4>(const size_t* idx, const StdLargeVec<Real>& indirect_indexed_data)
	{
		return
		{
			indirect_indexed_data[*idx],
			indirect_indexed_data[*(idx + 1)],
			indirect_indexed_data[*(idx + 2)],
			indirect_indexed_data[*(idx + 3)]
		};
	}

	// Generalized template function for loading indirect indexed data into a vector of batches
	// (alternative to xsimd::batch<T,A>::gather() functions)
	template< int /*number of elements in a batch*/, class ContainerType>
	VecdBatchSph LoadVecdBatchSph(const size_t* /*idx*/, const StdLargeVec<ContainerType>& /*indirect_indexed_data*/)
	{
		return{};
	}

	// Specialization for a two dimensional vector of batches, each packed with 4 values
	template<>
	inline VecdBatchSph LoadVecdBatchSph<4>(const size_t* idx, const StdLargeVec<Vecd>& indirect_indexed_data)
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

	// Generalized template function for loading direct indexed data into a single batch
	template< int /*number of elements in a batch*/, class ContainerType>
	SingleBatchSph LoadDirectSingleBatchSph(size_t idx, const StdLargeVec<ContainerType>& /*direct_indexed_data*/)
	{
		return{};
	}

	// Specialization for a batch packed with 4 values
	template<>
	inline SingleBatchSph LoadDirectSingleBatchSph<4>(size_t idx, const StdLargeVec<Real>& direct_indexed_data)
	{
		return
		{
			direct_indexed_data[idx],
			direct_indexed_data[idx+1],
			direct_indexed_data[idx+2],
			direct_indexed_data[idx+3]
		};
	}

	// Generalized template function for loading direct indexed data into a vector of batches
	template< int /*number of elements in a batch*/, class ContainerType>
	VecdBatchSph LoadDirectVecdBatchSph(size_t /*idx*/, const StdLargeVec<ContainerType>& /*direct_indexed_data*/)
	{
		return{};
	}

	// Specialization for a two dimensional vector of batches, each packed with 4 values
	template<>
	inline VecdBatchSph LoadDirectVecdBatchSph<4>(size_t idx, const StdLargeVec<Vecd>& direct_indexed_data)
	{
		return
		{
			{	// Batch X
				direct_indexed_data[idx][0],
				direct_indexed_data[idx+1][0],
				direct_indexed_data[idx+2][0],
				direct_indexed_data[idx+3][0]
			},
			{	// Batch Y
				direct_indexed_data[idx][1],
				direct_indexed_data[idx+1][1],
				direct_indexed_data[idx+2][1],
				direct_indexed_data[idx+3][1]
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
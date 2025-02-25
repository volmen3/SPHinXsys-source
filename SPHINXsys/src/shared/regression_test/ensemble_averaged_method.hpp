/**
 * @file 	ensemble_averaged_method.cpp
 * @author	Bo Zhang and Xiangyu Hu
 */

#pragma once

#include "ensemble_averaged_method.h"

 //=================================================================================================//
namespace SPH
{
	//=================================================================================================//
	template<class ObserveMethodType>
	void RegressionTestEnsembleAveraged<ObserveMethodType>::calculateNewVariance(TripleVec<Real> &result,
		DoubleVec<Real> &meanvalue_new, DoubleVec<Real> &variance, DoubleVec<Real> &variance_new)
	{
		for (int snapshot_index = 0; snapshot_index != SMIN(this->snapshot_, this->number_of_snapshot_old_); ++snapshot_index)
			for (int observation_index = 0; observation_index != this->observation_; ++observation_index)
				for (int run_index = 0; run_index != this->number_of_run_; ++run_index)
					variance_new[snapshot_index][observation_index] = SMAX(variance[snapshot_index][observation_index], variance_new[snapshot_index][observation_index],
						std::pow((result[run_index][snapshot_index][observation_index] - meanvalue_new[snapshot_index][observation_index]), 2), std::pow(meanvalue_new[snapshot_index][observation_index] * 1.0e-2, 2));
	};
	//=================================================================================================//
	template<class ObserveMethodType>
	void RegressionTestEnsembleAveraged<ObserveMethodType>::calculateNewVariance(TripleVec<Vecd> &result, 
		DoubleVec<Vecd> &meanvalue_new, DoubleVec<Vecd> &variance, DoubleVec<Vecd> &variance_new)
	{
		for (int snapshot_index = 0; snapshot_index != SMIN(this->snapshot_, this->number_of_snapshot_old_); ++snapshot_index) 
			for (int observation_index = 0; observation_index != this->observation_; ++observation_index) 
				for (int run_index = 0; run_index != this->number_of_run_; ++run_index) 
					for (int dimension_index = 0; dimension_index != variance[0][0].size(); ++dimension_index) 
						variance_new[snapshot_index][observation_index][dimension_index] = SMAX(variance[snapshot_index][observation_index][dimension_index], variance_new[snapshot_index][observation_index][dimension_index],
							std::pow((result[run_index][snapshot_index][observation_index][dimension_index] - meanvalue_new[snapshot_index][observation_index][dimension_index]), 2),
							std::pow(meanvalue_new[snapshot_index][observation_index][dimension_index] * 1.0e-2, 2));
	};
	//=================================================================================================//
	template<class ObserveMethodType>
	void RegressionTestEnsembleAveraged<ObserveMethodType>::calculateNewVariance(TripleVec<Matd> &result,
		DoubleVec<Matd> &meanvalue_new, DoubleVec<Matd> &variance, DoubleVec<Matd> &variance_new)
	{
		for (int snapshot_index = 0; snapshot_index != SMIN(this->snapshot_, this->number_of_snapshot_old_); ++snapshot_index)
			for (int observation_index = 0; observation_index != this->observation_; ++observation_index)
				for (int run_index = 0; run_index != this->number_of_run_; ++run_index)
					for (size_t dimension_index_i = 0; dimension_index_i != variance[0][0].size(); ++dimension_index_i)
						for (size_t dimension_index_j = 0; dimension_index_j != variance[0][0].size(); ++dimension_index_j)
							variance_new[snapshot_index][observation_index][dimension_index_i][dimension_index_j] = SMAX(variance[snapshot_index][observation_index][dimension_index_i][dimension_index_j], variance_new[snapshot_index][observation_index][dimension_index_i][dimension_index_j],
								std::pow((result[run_index][snapshot_index][observation_index][dimension_index_i][dimension_index_j] - meanvalue_new[snapshot_index][observation_index][dimension_index_i][dimension_index_j]), 2),
								std::pow(meanvalue_new[snapshot_index][observation_index][dimension_index_i][dimension_index_j] * 1.0e-2, 2));
	};
	//=================================================================================================//
	template<class ObserveMethodType>
	int RegressionTestEnsembleAveraged<ObserveMethodType>::compareParameter(string par_name, 
		DoubleVec<Real> &parameter, DoubleVec<Real> &parameter_new, Real &threshold)
	{
		int count = 0;
		for (int snapshot_index = 0; snapshot_index != SMIN(this->snapshot_, this->number_of_snapshot_old_); ++snapshot_index) 
			for (int observation_index = 0; observation_index != this->observation_; ++observation_index)
			{
				Real relative_value_ = ABS((parameter[snapshot_index][observation_index] - parameter_new[snapshot_index][observation_index]) / (parameter_new[snapshot_index][observation_index] + TinyReal));
				if (relative_value_ > threshold)
				{
					std::cout << par_name << ": " << this->quantity_name_ << "[" << observation_index << "] in " << this->element_tag_[snapshot_index]
						<< " is not converged, and difference is " << relative_value_ << endl;
					count++;
				}
			}	
		return count;
	};
	////=================================================================================================//
	template<class ObserveMethodType>
	int RegressionTestEnsembleAveraged<ObserveMethodType>::compareParameter(string par_name,
		DoubleVec<Vecd> &parameter, DoubleVec<Vecd> &parameter_new, Vecd &threshold)
	{
		int count = 0;
		for (int snapshot_index = 0; snapshot_index != SMIN(this->snapshot_, this->number_of_snapshot_old_); ++snapshot_index) 
			for (int observation_index = 0; observation_index != this->observation_; ++observation_index) 
				for (int dimension_index = 0; dimension_index != parameter[0][0].size(); ++dimension_index)
				{
					Real relative_value_ = ABS((parameter[snapshot_index][observation_index][dimension_index] - parameter_new[snapshot_index][observation_index][dimension_index]) / (parameter_new[snapshot_index][observation_index][dimension_index] + TinyReal));
					if (relative_value_ > threshold[dimension_index])
					{
						std::cout << par_name << ": " << this->quantity_name_ << "[" << observation_index << "][" << dimension_index << "] in " << this->element_tag_[snapshot_index]
							<< " is not converged, and difference is " << relative_value_ << endl;
						count++;
					}
				}
		return count;
	};
	////=================================================================================================// 
	template<class ObserveMethodType>
	int RegressionTestEnsembleAveraged<ObserveMethodType>::compareParameter(string par_name,
		DoubleVec<Matd> &parameter, DoubleVec<Matd> &parameter_new, Matd &threshold)
	{
		int count = 0;
		for (int snapshot_index = 0; snapshot_index != SMIN(this->snapshot_, this->number_of_snapshot_old_); ++snapshot_index)
			for (int observation_index = 0; observation_index != this->observation_; ++observation_index)
				for (int dimension_index_i = 0; dimension_index_i != parameter[0][0].size(); ++dimension_index_i)
					for (int dimension_index_j = 0; dimension_index_j != parameter[0][0].size(); ++dimension_index_j)
					{
						Real relative_value_ = ABS(parameter[snapshot_index][observation_index][dimension_index_i][dimension_index_j] - parameter_new[snapshot_index][observation_index][dimension_index_i][dimension_index_j])
							/ (parameter_new[snapshot_index][observation_index][dimension_index_i][dimension_index_j] + TinyReal);
						if (relative_value_ > threshold[dimension_index_i][dimension_index_j])
						{
							std::cout << par_name << ": " << this->quantity_name_ << "[" << observation_index << "][" << dimension_index_i << "][" << dimension_index_j << " ] in "
								<< this->element_tag_[snapshot_index] << " is not converged, and difference is " << relative_value_ << endl;
							count++;
						}
					}
		return count;
	};
	//=================================================================================================//
	template<class ObserveMethodType>
	int RegressionTestEnsembleAveraged<ObserveMethodType>::testNewResult(int diff, DoubleVec<Real> &current_result,
		DoubleVec<Real> &meanvalue, DoubleVec<Real> &variance)
	{
		int count = 0;
		for (int snapshot_index = 0; snapshot_index != SMIN(this->snapshot_, this->number_of_snapshot_old_); ++snapshot_index)
		{
			for (int observation_index = 0; observation_index != this->observation_; ++observation_index)
			{
				Real relative_value_ = (std::pow(current_result[snapshot_index][observation_index] - meanvalue[snapshot_index + diff][observation_index], 2) - variance[snapshot_index + diff][observation_index]) / variance[snapshot_index + diff][observation_index];
				if (relative_value_ > 0.01)
				{
					std::cout << this->quantity_name_ << "[" << observation_index << "] in " << this->element_tag_[snapshot_index] << " is beyond the exception, and difference is "
						<< relative_value_ << endl;
					count++;
				}
			}
		}
		return count;
	};
	//=================================================================================================//
	template<class ObserveMethodType>
	int RegressionTestEnsembleAveraged<ObserveMethodType>::testNewResult(int diff, DoubleVec<Vecd> &current_result,
		DoubleVec<Vecd> &meanvalue, DoubleVec<Vecd> &variance)
	{
		int count = 0;
		for (int snapshot_index = 0; snapshot_index != SMIN(this->snapshot_, this->number_of_snapshot_old_); ++snapshot_index)
		{
			for (int observation_index = 0; observation_index != this->observation_; ++observation_index)
			{
				for (int dimension_index = 0; dimension_index != meanvalue[0][0].size(); ++dimension_index)
				{
					Real relative_value_ = (std::pow(current_result[snapshot_index][observation_index][dimension_index] - meanvalue[snapshot_index + diff][observation_index][dimension_index], 2) - variance[snapshot_index + diff][observation_index][dimension_index]) / variance[snapshot_index + diff][observation_index][dimension_index];
					if (relative_value_ > 0.01)
					{
						std::cout << this->quantity_name_ << "[" << observation_index << "][" << dimension_index << "] in " << this->element_tag_[snapshot_index] << " is beyond the exception, and difference is "
							<< relative_value_ << endl;
						count++;
					}
				}
			}	
		}
		return count;
	};
	//=================================================================================================// 
	template<class ObserveMethodType>
	int RegressionTestEnsembleAveraged<ObserveMethodType>::testNewResult(int diff, DoubleVec<Matd> &current_result,
		DoubleVec<Matd> &meanvalue, DoubleVec<Matd> &variance)
	{
		int count = 0;
		std::cout << "The current length difference is " << diff << "." << endl;
		for (int snapshot_index = 0; snapshot_index != SMIN(this->snapshot_, this->number_of_snapshot_old_); ++snapshot_index)
		{
			for (int observation_index = 0; observation_index != this->observation_; ++observation_index)
			{
				for (int dimension_index_i = 0; dimension_index_i != meanvalue[0][0].size(); ++dimension_index_i)
				{
					for (int dimension_index_j = 0; dimension_index_j != meanvalue[0][0].size(); ++dimension_index_j)
					{
						Real relative_value_ = (std::pow(current_result[snapshot_index][observation_index][dimension_index_i][dimension_index_j] - meanvalue[snapshot_index + diff][observation_index][dimension_index_i][dimension_index_j], 2) - variance[snapshot_index + diff][observation_index][dimension_index_i][dimension_index_j]) / variance[snapshot_index + diff][observation_index][dimension_index_i][dimension_index_j];
						if (relative_value_ > 0.01)
						{
							std::cout << this->quantity_name_ << "[" << observation_index << "][" << dimension_index_i << "] in " << this->element_tag_[snapshot_index] << " is beyond the exception, and difference is "
								<< relative_value_ << endl;
							count++;
						}
					}
				}
			}	
		}
		return count;
	};
	//=================================================================================================//
	template<class ObserveMethodType>
	void RegressionTestEnsembleAveraged<ObserveMethodType>::setupAndCorrection()
	{
		this->snapshot_ = this->current_result_.size();
		this->observation_ = this->current_result_[0].size();

		if (this->number_of_run_ > 1)
		{
			if (this->converged == "false" ) /*< To identify the database generation or new result testing. */
			{
				if (!fs::exists(this->result_filefullpath_))
				{
					std::cout << "\n Error: the input file:" << this->result_filefullpath_ << " is not exists" << std::endl;
					std::cout << __FILE__ << ':' << __LINE__ << std::endl;
					exit(1);
				}
				else
					this->result_xml_engine_in_.loadXmlFile(this->result_filefullpath_);
			}

			if (!fs::exists(this->mean_variance_filefullpath_))
			{
				std::cout << "\n Error: the input file:" << this->mean_variance_filefullpath_ << " is not exists" << std::endl;
				std::cout << __FILE__ << ':' << __LINE__ << std::endl;
				exit(1);
			}
			else
			{
				this->mean_variance_xml_engine_in_.loadXmlFile(this->mean_variance_filefullpath_);
				SimTK::Xml::Element mean_element_ = this->mean_variance_xml_engine_in_.getChildElement("Mean_Element");
				this->number_of_snapshot_old_ = std::distance(mean_element_.element_begin(), mean_element_.element_end());

				DoubleVec<VariableType> temp(SMAX(this->snapshot_, this->number_of_snapshot_old_), StdVec<VariableType>(this->observation_));
				meanvalue_ = temp;
				variance_ = temp;

				/** Unify the length of current result and previous result. */
				if (this->number_of_snapshot_old_ < this->snapshot_)
				{
					this->difference_ = this->snapshot_ - this->number_of_snapshot_old_;
					for (int delete_ = 0; delete_ != this->difference_; ++delete_)
						this->current_result_.pop_back();
				}
				else if (this->number_of_snapshot_old_ > this->snapshot_)
					this->difference_ = this->number_of_snapshot_old_ - this->snapshot_;
				else
					this->difference_ = 0;
			}
		}
		else if (this->number_of_run_ == 1)
		{
			this->number_of_snapshot_old_ = this->snapshot_;
			DoubleVec<VariableType> temp(this->snapshot_, StdVec<VariableType>(this->observation_));
			this->result_.push_back(this->current_result_);
			meanvalue_ = temp;
			variance_ = temp;
		}
	};
	//=================================================================================================//
	template<class ObserveMethodType>
	void RegressionTestEnsembleAveraged<ObserveMethodType>::readMeanVarianceFromXml()
	{
		if (this->number_of_run_ > 1)
		{
			SimTK::Xml::Element mean_element_ = this->mean_variance_xml_engine_in_.getChildElement("Mean_Element");
			SimTK::Xml::Element variance_element_ = this->mean_variance_xml_engine_in_.getChildElement("Variance_Element");
			for (int observation_index = 0; observation_index != this->observation_; ++observation_index)
			{
				this->xmlmemory_io_.readDataFromXmlMemory(this->mean_variance_xml_engine_in_, 
					mean_element_, observation_index, this->meanvalue_, this->quantity_name_);
				this->xmlmemory_io_.readDataFromXmlMemory(this->mean_variance_xml_engine_in_,
					variance_element_, observation_index, this->variance_, this->quantity_name_);
			}
		}
	};
	//=================================================================================================//
	template<class ObserveMethodType>
	void RegressionTestEnsembleAveraged<ObserveMethodType>::updateMeanVariance()
	{
		/** Unify the length of result and meanvalue. */
		if (this->number_of_run_ > 1)
		{
			for (int delete_ = 0; delete_ != this->difference_; ++delete_)
			{
				meanvalue_.pop_back(); 
				variance_.pop_back();
			}
		}
		meanvalue_new_ = meanvalue_;
		variance_new_ = variance_;

		/** update the meanvalue of the result. */
		for (int snapshot_index = 0; snapshot_index != SMIN(this->snapshot_, this->number_of_snapshot_old_); ++snapshot_index)
			for (int observation_index = 0; observation_index != this->observation_; ++observation_index)
				meanvalue_new_[snapshot_index][observation_index] = (meanvalue_[snapshot_index][observation_index] * (this->number_of_run_ - 1) + this->current_result_[snapshot_index][observation_index]) / this->number_of_run_;
		/** Update the variance of the result. */
		calculateNewVariance(this->result_, meanvalue_new_, variance_, variance_new_);
	};
	//=================================================================================================//
	template<class ObserveMethodType>
	void RegressionTestEnsembleAveraged<ObserveMethodType>::writeMeanVarianceToXml()
	{
		this->mean_variance_xml_engine_out_.addElementToXmlDoc("Mean_Element");
		SimTK::Xml::Element mean_element_ = this->mean_variance_xml_engine_out_.getChildElement("Mean_Element");
		this->xmlmemory_io_.writeDataToXmlMemory(this->mean_variance_xml_engine_out_, mean_element_, this->meanvalue_new_,
			SMIN(this->snapshot_, this->number_of_snapshot_old_), this->observation_, this->quantity_name_, this->element_tag_);
		this->mean_variance_xml_engine_out_.addElementToXmlDoc("Variance_Element");
		SimTK::Xml::Element variance_element_ = this->mean_variance_xml_engine_out_.getChildElement("Variance_Element");
		this->xmlmemory_io_.writeDataToXmlMemory(this->mean_variance_xml_engine_out_, variance_element_, this->variance_new_,
			SMIN(this->snapshot_, this->number_of_snapshot_old_), this->observation_, this->quantity_name_, this->element_tag_);
		this->mean_variance_xml_engine_out_.writeToXmlFile(this->mean_variance_filefullpath_);
	};
	//=================================================================================================//
	template<class ObserveMethodType>
	bool RegressionTestEnsembleAveraged<ObserveMethodType>::compareMeanVariance()
	{
		int count_not_converged_m = 0;
		int count_not_converged_v = 0;
		count_not_converged_m = compareParameter("meanvalue", meanvalue_, meanvalue_new_, this->threshold_mean_);
		count_not_converged_v = compareParameter("variance", variance_, variance_new_, this->threshold_variance_);
		if (count_not_converged_m == 0)
		{
			std::cout << "The meanvalue of " << this->quantity_name_ << " are converged now." << endl;
			if (count_not_converged_v == 0)
			{
				if (this->label_for_repeat_ == 4)
				{
					this->converged = "true";
					this->label_for_repeat_++;
					std::cout << "The meanvalue and variance of " << this->quantity_name_ << " are converged enough times, and run will stop now." << endl;
					return true;
				}
				else
				{
					this->converged = "false";
					this->label_for_repeat_++;
					std::cout << "The variance of " << this->quantity_name_ << " are also converged, and this is the " << this->label_for_repeat_
						<< " times. They should be converged more times to be stable." << endl;
					return false;
				}
			}
			else if (count_not_converged_v != 0)
			{
				this->converged = "false";
				this->label_for_repeat_ = 0;
				std::cout << "The variance of " << this->quantity_name_ << " are not converged " << count_not_converged_v << " times." << endl;
				return false;
			};
		}
		else if (count_not_converged_m != 0)
		{
			this->converged = "false";
			this->label_for_repeat_ = 0;
			std::cout << "The meanvalue of " << this->quantity_name_ << " are not converged " << count_not_converged_m << " times." << endl;
			return false;
		}
	};
	//=================================================================================================//
	template<class ObserveMethodType>
	void RegressionTestEnsembleAveraged<ObserveMethodType>::resultTest()
	{
		/* compare the current result to the converged mean value and variance. */
		int test_wrong = 0;
		if (this->snapshot_ < this->number_of_snapshot_old_)
			test_wrong = testNewResult(this->difference_, this->current_result_, meanvalue_, variance_);
		else
		{
			/** Unify the length of meanvalue, variance, old result, new result. */
			for (int delete_ = 0; delete_ != this->difference_; ++delete_)
			{
				meanvalue_.pop_back();
				variance_.pop_back();
			}
			test_wrong = testNewResult(0, this->current_result_, meanvalue_, variance_);
		}
		/* draw the conclusion. */
		if (test_wrong == 0)
			std::cout << "The result of " << this->quantity_name_ << " are correct based on the ensemble averaged regression test!" << endl;
		else
		{
			std::cout << "There are " << test_wrong << " snapshots are not within the expected range." << endl;
			std::cout << "Please try again. If it still post this conclusion, the result is not correct!" << endl;
			exit(1);
		}
	};
	//=================================================================================================//
}
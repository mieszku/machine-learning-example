#include <iostream>
#include <cassert>

#include "neuron.hxx"

namespace ml
{
	namespace
	{
		const value_type DEFAULT_VALUE = 1;

		static void
		assert_connection(neuron *__in,
						  neuron *__out)
		{
			assert(
				(__in->get_type() == neuron::TYPE_BIAS &&
				__out->get_type() != neuron::TYPE_BIAS) ||

				(__in->get_type() == neuron::TYPE_INPUT &&
				__out->get_type() == neuron::TYPE_HIDDEN) ||

				(__in->get_type() == neuron::TYPE_INPUT &&
				__out->get_type() == neuron::TYPE_OUTPUT) ||

				(__in->get_type() == neuron::TYPE_HIDDEN &&
				__out->get_type() == neuron::TYPE_OUTPUT)
			);
		}
	}



	neuron::neuron(neural_net *__net,
				   type		   __type)
		: _M_net(__net)
		, _M_type(__type)
		, _M_value(DEFAULT_VALUE)
	{
	}

	neuron::~neuron()
	{
	}

	void
	neuron::connect_input(neuron 	 	*__input,
						  shared_weight  __weight)
	{
		assert_connection(__input, this);

		connection __conn(__input, __weight);

		_M_inputs.push_back(__conn);
	}

	void
	neuron::connect_output(neuron 		 *__output,
						   shared_weight  __weight)
	{
		assert_connection(this, __output);

		connection __conn(__output, __weight);

		_M_outputs.push_back(__conn);
	}

	void
	neuron::recalculate()
	{
		value_type __value = 0;

		for (auto &__input : _M_inputs)
			__value += __input.weight() * __input.get_neuron()->value();

		value() = __value;
	}
}

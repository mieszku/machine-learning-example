/*
 * Created by Mieszko Mazurek <mimaz@gmx.com>
 * April 2017
 */

#include <iostream>
#include <cassert>

#include "neuron.hxx"
#include "neural-net.hxx"

namespace ml
{
	namespace
	{
		const value_type DEFAULT_VALUE = 1;

		static void 
		assert_connection(neuron *__in,
						  neuron *__out)
		{
			switch (__in->get_type())
			{
			case neuron::TYPE_INPUT:
				assert(__out->get_type() == neuron::TYPE_HIDDEN);
				break;

			case neuron::TYPE_HIDDEN:
				assert(__out->get_type() == neuron::TYPE_HIDDEN ||
					   __out->get_type() == neuron::TYPE_OUTPUT);
				break;

			case neuron::TYPE_OUTPUT:
				assert(false);
				break;

			case neuron::TYPE_BIAS:
				assert(__out->get_type() != neuron::TYPE_BIAS);
			}
		}
	}



	neuron::neuron(neural_net *__net,
				   type		   __type)
		: _M_net(__net)
		, _M_type(__type)
		, _M_value(DEFAULT_VALUE)
		, _M_gradient(0)
	{}

	neuron::~neuron()
	{}

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

		value() = transfer(__value);
	}

	void
	neuron::update_output_gradient(value_type __target)
	{
		value_type __delta = __target - value();
		gradient() = __delta * transfer_derivative(value());
	}

	void
	neuron::update_gradient()
	{
		value_type __sum = 0;

		for (auto &__conn : _M_outputs)
			__sum += __conn.weight() * __conn.get_neuron()->gradient();

		gradient() = __sum * transfer_derivative(value());
	}

	void
	neuron::update_weights()
	{
		const value_type ETA = 0.05;
		const value_type ALPHA = 0.9;


		for (auto &__conn : _M_inputs)
		{
			value_type __old = __conn.weight_delta();

			value_type __new = ETA * __conn.get_neuron()->value() * gradient() 
							 + ALPHA * __conn.weight_delta();

			__conn.weight_delta() = __new;
			__conn.weight() += __new;
		}
	}
}

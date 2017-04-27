/*
 * Created by Mieszko Mazurek <mimaz@gmx.com>
 * April 2017
 */

#include <iostream>
#include <cassert>
#include <cmath>

#include "neural-net.hxx"

namespace ml
{
	namespace
	{
		static neuron::shared_weight
		make_shared_weight()
		{
			auto __weight = std::make_shared<neuron::weight_type>();

			__weight->value = (value_type) rand() / RAND_MAX;
			__weight->delta = 0;

			return __weight;
		}

		static void
		connect_neurons(neuron *__in,
						neuron *__out)
		{
			auto __shared_weight = make_shared_weight();

			__in->connect_output(__out, __shared_weight);
			__out->connect_input(__in, __shared_weight);
		}
	}




	value_type
	transfer(value_type __value)
	{
		return tanh(__value);
	}

	value_type
	transfer_derivative(value_type __value)
	{
		value_type __normal = tanh(__value);

		return 1.0 - __normal * __normal;
	}



	neural_net::neural_net(int *__topo_ptr,
						   int  __layers)
	{
		assert(__layers > 1);

		_M_bias = new neuron(this, neuron::TYPE_BIAS);

		_M_neurons.resize(__layers);


		layer *__prev_layer = nullptr;

		auto init_layer = [this, &__topo_ptr, &__prev_layer]
						  (layer 		&__layer,
						   neuron::type  __type) -> void 
		{
			__layer = layer(*__topo_ptr++);

			for (auto &__neuron : __layer)
			{
				__neuron = new neuron(this, __type);

				connect_neurons(get_bias(), __neuron);
			}

			if (__prev_layer != nullptr)
				for (auto __neuron : __layer)
					for (auto __prev : *__prev_layer)
						connect_neurons(__prev, __neuron);

			__prev_layer = &__layer;
		};



		init_layer(_M_neurons.front(), neuron::TYPE_INPUT);

		for (int __i = 1; __i < __layers - 1; __i++)
			init_layer(_M_neurons[__i], neuron::TYPE_HIDDEN);

		init_layer(_M_neurons.back(), neuron::TYPE_OUTPUT);
	}

	neural_net::~neural_net()
	{
		for (auto &__layer : _M_neurons)
			for (auto __neuron : __layer)
				delete __neuron;

		delete _M_bias;
	}

	void
	neural_net::set_input_data(const value_type *__data,
							   int		  	   	 __length)
	{
		assert(_M_neurons.front().size() == __length);


		auto &__layer = _M_neurons.front();

		auto __data_it = __data;

		for (auto __neuron : __layer)
			__neuron->value() = *__data++;
	}

	void
	neural_net::recalculate()
	{
		for (int __i = 1; __i < (int) _M_neurons.size(); __i++)
			for (auto __neuron : _M_neurons[__i])
				__neuron->recalculate();
	}

	void
	neural_net::propagate_back(const value_type *__data,
							   int				 __length)
	{
		assert(__length == _M_neurons.back().size());



		auto &__output_layer = _M_neurons.back();

		auto __data_it = __data;


		for (auto __neuron : __output_layer)
			__neuron->update_output_gradient(*__data_it++);


		for (int __i = _M_neurons.size() - 2; __i > 0; __i--)
			for (auto __neuron : _M_neurons[__i])
				__neuron->update_gradient();



		for (int __i = 1; __i < _M_neurons.size(); __i++)
			for (auto __neuron : _M_neurons[__i])
				__neuron->update_weights();
	}


	void
	neural_net::train(const value_type *__input,
					  int 				__input_length,
					  const value_type *__output,
					  int				__output_length)
	{
		set_input_data(__input, __input_length);
		recalculate();

		propagate_back(__output, __output_length);
		recalculate();
	}

	void
	neural_net::get_result(value_type *__data,
						   int		   __length) const
	{
		auto __it = __data;

		for (auto __neuron : _M_neurons.back())
			*__it++ = __neuron->value();
	}

	value_type
	neural_net::get_result(int __index) const
	{
		assert(__index < _M_neurons.back().size());

		return _M_neurons.back()[__index]->value();
	}
}

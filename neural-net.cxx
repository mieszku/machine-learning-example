#include <iostream>
#include <cassert>

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
	neural_net::feed_forward(const value_type *__data,
							 int		  	   __length)
	{
		{
			auto &__layer = _M_neurons.front();

			assert(__layer.size() == __length);


			for (auto __neuron : __layer)
				__neuron->value() = *__data++;
		}


		for (int __i = 1; __i < (int) _M_neurons.size(); __i++)
			for (auto __neuron : _M_neurons[__i])
				__neuron->recalculate();
	}
}

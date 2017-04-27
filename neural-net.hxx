/*
 * Created by Mieszko Mazurek <mimaz@gmx.com>
 * April 2017
 */

#ifndef __neural_net__
#define __neural_net__

#include <vector>

#include "neuron.hxx"

namespace ml
{
	value_type transfer(value_type __value);
	value_type transfer_derivative(value_type __value);



	class neural_net
	{
	public:
		typedef std::vector<neuron *> layer;

		neural_net(int *__topo, 
				   int  __layers);
		neural_net(const neural_net &) = delete;
		neural_net(neural_net &&) = default;

		~neural_net();

		void set_input_data(const value_type *__data,
						  	int		   		  __length);

		void recalculate();

		void propagate_back(const value_type *__data,
						    int				  __length);

		void train(const value_type *__input,
				   int				 __input_length,
				   const value_type *__output,
				   int				 __output_length);

		void get_result(value_type *__data,
						int	   		__length) const;

		value_type get_result(int __index) const;

		neuron *get_bias() const { return _M_bias; }

	private:
		std::vector<layer>  _M_neurons;
		neuron 			   *_M_bias;
	};
}

#endif

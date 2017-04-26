#ifndef __neural_net__
#define __neural_net__

#include <vector>

#include "neuron.hxx"

namespace ml
{
	class neural_net
	{
	public:
		typedef std::vector<neuron *> layer;

		neural_net(int *__topo, 
				   int  __layers);
		neural_net(const neural_net &) = delete;
		neural_net(neural_net &&) = default;

		~neural_net();

		void feed_forward(const float *__data,
						  int		   __length);

		neuron *get_bias() const { return _M_bias; }

	private:
		std::vector<layer>  _M_neurons;
		neuron 			   *_M_bias;
	};
}

#endif

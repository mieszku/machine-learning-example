#ifndef __neuron_hxx__
#define __neuron_hxx__

#include <memory>
#include <vector>

namespace ml
{
	typedef float value_type;

	class neural_net;

	class neuron
	{
	public:
		class weight_type : private std::pair<value_type, value_type>
		{
		public:
			value_type &value = first;
			value_type &delta = second;
		};

		typedef std::shared_ptr<weight_type> shared_weight;



		class connection
		{
		public:
			connection(neuron 		 *__neuron,
					   shared_weight  __shared_weight)
					: _M_neuron(__neuron)
					, _M_shared_weight(__shared_weight) {}

			ml::neuron *get_neuron() const { return _M_neuron; }
			value_type &weight() const { return _M_shared_weight->value; }
			value_type &weight_delta() const { return _M_shared_weight->delta; }

		private:
			ml::neuron 	  *_M_neuron;
			shared_weight  _M_shared_weight;
		};



		enum type 
		{
			TYPE_INPUT,
			TYPE_HIDDEN,
			TYPE_OUTPUT,
			TYPE_BIAS
		};



		neuron(neural_net *__net,
			   type		   __type);

		neuron(const neuron &) = delete;
		neuron(neuron &&) = delete;

		~neuron();

		void connect_input(neuron 		 *__input, 
						   shared_weight  __weight);

		void connect_output(neuron 		  *__output,
							shared_weight  __weight);

		void recalculate();

		neural_net  *get_net() const { return _M_net; }
		type 		 get_type() const { return _M_type; }
		value_type  &value() { return _M_value; }
		
	private:
		neural_net 	  	 *const _M_net;
		const type			 	_M_type;
		value_type 			 	_M_value;
		std::vector<connection>	_M_inputs;
		std::vector<connection> _M_outputs;
	};
}

#endif

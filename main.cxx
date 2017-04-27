/*
 * Created by Mieszko Mazurek <mimaz@gmx.com>
 * April 2017
 */

#include <iostream>
#include <vector>

#include "neural-net.hxx"

int main (int argc, char **argv)
{
	(void) argc;
	(void) argv;

	/*
	 * neural net topology:
	 * count of neurons from input to output layer
	 *
	 * here two input neurons, then two hidden layers
	 * with 9 and 5 neurons in order and one output
	 *
	 * smaller net may need more epochs of training
	 */
	int topo[] = { 2, 6, 3, 1 };
	int layer_count = sizeof(topo) / sizeof(topo[0]);

	/*
	 * our net instance
	 */
	ml::neural_net net(topo, layer_count);

	/*
	 * data buffers
	 */
	float in[2];
	float out[1];

	/*
	 * try to manipulate number of epochs,
	 * doing less count of training results less (or no)
	 * prediction efficiency
	 *
	 * by doing more trainings (i.e. 1000 or more) you can 
	 * find that net answers quite precisely
	 */
	for (int i = 0; i < 1000; i++)
	{
		/*
		 * generating some training data:
		 *
		 * get two random bits and XOR it to the r variable
		 */
		int p = rand() & 1;
		int q = rand() & 1;
		int r = p ^ q;

		/*
		 * put data into buffers
		 */
		in[0] = p;
		in[1] = q;
		out[0] = r;

		/*
		 * train our neural net
		 */
		net.train(in, 2, out, 1);

		/*
		 * get answer calculated at the end of training
		 * the argument indicates index of output unit,
		 * we have one output neuron so it's 0
		 */
		float answer = net.get_result(0);

		/*
		 * print input data, expected result, answer and error
		 */
		std::cout << std::fixed
				  << "expecting: " << r 
				  << ", answer: " << net.get_result(0) 
				  << ", error: " << (net.get_result(0) - r) 
				  << std::endl;
	}

	return 0;
}

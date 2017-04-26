#include <iostream>
#include <vector>

#include "neuron.hxx"
#include "neural-net.hxx"

int main (int argc, char **argv)
{
	(void) argc;
	(void) argv;

	int topo[] = { 2, 3, 2 };
	ml::neural_net __net(topo, 3);

	float food[] = { 1, 0 };
	__net.feed_forward(food, 2);

	return 0;
}

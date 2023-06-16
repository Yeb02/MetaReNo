#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <tuple>
#include <fstream>

#include "Random.h"
#include "Config.h"

#include "InternalConnexion_G.h"



// Util:
inline int binarySearch(std::vector<float>& proba, float value) {
	int inf = 0;
	int sup = (int)proba.size() - 1;

	if (proba[inf] > value) {
		return inf;
	}

	int mid;
	int max_iter = 15;
	while (sup - inf >= 1 && max_iter--) {
		mid = (sup + inf) / 2;
		if (proba[mid] < value && value <= proba[mid + 1]) {
			return mid + 1;
		}
		else if (proba[mid] < value) {
			inf = mid;
		}
		else {
			sup = mid;
		}
	}
	return 0; // not necessarily a failure, since floating point approximation prevents the sum from reaching 1.
	//throw "Binary search failure !";
}

inline int binarySearch(float* proba, float value, int size) {
	int inf = 0;
	int sup = size - 1;

	if (proba[inf] > value) {
		return inf;
	}

	int mid;
	int max_iter = 15;
	while (sup - inf >= 1 && max_iter--) {
		mid = (sup + inf) / 2;
		if (proba[mid] < value && value <= proba[mid + 1]) {
			return mid + 1;
		}
		else if (proba[mid] < value) {
			inf = mid;
		}
		else {
			sup = mid;
		}
	}
	return 0; // not necessarily a failure, since floating point approximation prevents the sum from reaching 1.
	//throw "Binary search failure !";
}

struct ComplexNode_G {

	// Does not do much, because most attributes are set by the network owning this.
	ComplexNode_G(int inputSize, int outputSize);

	// WARNING ! "this" node is now a deep copy of n, but the pointers towards the children 
	// must be updated manually if "this" and n do not belong to the same Network !
	// (typically in Network(Network * n))
	ComplexNode_G(ComplexNode_G* n);

	~ComplexNode_G() {};

	ComplexNode_G(std::ifstream& is);
	void save(std::ofstream& os);

	int inputSize, outputSize; // >= 1

	
	// Contains pointers to the children. A pointer can appear multiple times.
	std::vector<ComplexNode_G*> complexChildren;


	// Struct containing the constant, evolved, matrix of parameters linking internal nodes.
	// The name specifies the type of node that takes the result of the matrix operations as inputs.
	// nLines = sum(node.inputSize) for node of the type corresponding to the name
	// nColumns = this.inputSize + MODULATION_VECTOR_SIZE + sum(complexChild.inputSize) + sum(memoryChild.inputSize)
	InternalConnexion_G toComplex, toMemory, toModulation, toOutput;

	// The position in the genome vector. Must be genome.size() for the top node.
	int position;


	// Allocates and randomly initializes internal connexions.
	void createInternalConnexions();


	// Compute the size of the array containing the pre synaptic activations of the phenotype, preSynActs
	void computePreSynActArraySize(std::vector<int>& genomeState);

	// Compute the size of the array containing the post synaptic activations of the phenotype, postSynAct.
	void computePostSynActArraySize(std::vector<int>& genomeState);



#ifdef SATURATION_PENALIZING
	// Used to compute the size of the array containing the average saturations of the phenotype.
	void computeSaturationArraySize(std::vector<int>& genomeState);
#endif 

};


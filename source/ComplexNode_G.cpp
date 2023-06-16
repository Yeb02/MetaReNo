#include "ComplexNode_G.h"

ComplexNode_G::ComplexNode_G(int inputSize, int outputSize) :
	inputSize(inputSize), outputSize(outputSize),
	toComplex(0, 0),
	toMemory(0, 0),
	toModulation(0, 0),
	toOutput(0, 0)
{
	
	// The following initializations MUST be done outside.
	{
		position = -1;
	}
};

ComplexNode_G::ComplexNode_G(ComplexNode_G* n) {

	inputSize = n->inputSize;
	outputSize = n->outputSize;

	toComplex = n->toComplex;
	toMemory = n->toMemory;
	toModulation = n->toModulation;
	toOutput = n->toOutput;


	position = n->position;
	
	// The following enclosed section is useless if n is not part of the same network as "this", 
	// and it must be repeated where this function was called.
	{
		complexChildren.reserve((int)((float)n->complexChildren.size() * 1.5f));
		for (int j = 0; j < n->complexChildren.size(); j++) {
			complexChildren.emplace_back(n->complexChildren[j]);
		}
	}
}


ComplexNode_G::ComplexNode_G(std::ifstream& is) {
	// These must be set by the network in its Network(std::ifstream& is) constructor.
	{
		position = -1;
	}

	READ_4B(inputSize, is);
	READ_4B(outputSize, is);

	int _s;
	READ_4B(_s, is);
	complexChildren.resize(_s);


	toComplex = InternalConnexion_G(is);
	toMemory = InternalConnexion_G(is);
	toModulation = InternalConnexion_G(is);
	toOutput = InternalConnexion_G(is);

}

void ComplexNode_G::save(std::ofstream& os) {
	WRITE_4B(inputSize, os);
	WRITE_4B(outputSize, os);

	int _s;
	_s = (int)complexChildren.size();
	WRITE_4B(_s, os);


	toComplex.save(os);
	toMemory.save(os);
	toModulation.save(os);
	toOutput.save(os);
}


void ComplexNode_G::createInternalConnexions() {

	int nColumns = inputSize + MODULATION_VECTOR_SIZE;
	for (int i = 0; i < complexChildren.size(); i++) {
		nColumns += complexChildren[i]->outputSize;
	}


	int nLines;

	nLines = 0;
	for (int i = 0; i < complexChildren.size(); i++) {
		nLines += complexChildren[i]->inputSize;
	}
	toComplex = InternalConnexion_G(nLines, nColumns);

	nLines = outputSize;
	toOutput = InternalConnexion_G(nLines, nColumns);

	nLines = MODULATION_VECTOR_SIZE;
	toModulation = InternalConnexion_G(nLines, nColumns);
}


void ComplexNode_G::computePreSynActArraySize(std::vector<int>& genomeState) {
	int s = outputSize + MODULATION_VECTOR_SIZE;
	for (int i = 0; i < complexChildren.size(); i++) {
		s += complexChildren[i]->inputSize;
		if (genomeState[complexChildren[i]->position] == 0) {
			complexChildren[i]->computePreSynActArraySize(genomeState);
		}
		s += genomeState[complexChildren[i]->position];
	}

	genomeState[position] = s;
}

void ComplexNode_G::computePostSynActArraySize(std::vector<int>& genomeState) {
	int s = inputSize + MODULATION_VECTOR_SIZE;
	for (int i = 0; i < complexChildren.size(); i++) {
		s += complexChildren[i]->outputSize;
		if (genomeState[complexChildren[i]->position] == 0) {
			complexChildren[i]->computePostSynActArraySize(genomeState);
		}
		s += genomeState[complexChildren[i]->position];
	}

	genomeState[position] = s;
}

#ifdef SATURATION_PENALIZING
// Used to compute the size of the array containing the average saturations of the phenotype.
void ComplexNode_G::computeSaturationArraySize(std::vector<int>& genomeState) {
	int s = MODULATION_VECTOR_SIZE;
	for (int i = 0; i < complexChildren.size(); i++) {
		s += complexChildren[i]->inputSize;
		if (genomeState[complexChildren[i]->position] == 0) {
			complexChildren[i]->computeSaturationArraySize(genomeState);
		}
		s += genomeState[complexChildren[i]->position];
	}
	genomeState[position] = s;
}
#endif 




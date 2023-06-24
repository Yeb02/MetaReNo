#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <fstream>

#include "Random.h"
#include "ComplexNode_P.h"
#include "ComplexNode_G.h"



struct Network {

	Network(int inputSize, int outputSize);

	Network(Network* n);
	~Network() {};

	Network(std::ifstream& is);
	
	void save(std::ofstream& os);

	// Since getOutput returns a float*, application must either use it before any other call to step(),
	// destroyPhenotype(), preTrialReset(), ... or Network destruction, either deep copy the
	// pointee immediatly when getOutput() returns. If unsure, deep copy.
	float* getOutput();

	void step(const std::vector<float>& obs);
	
	void createPhenotype();
	void destroyPhenotype();

	// Sets to 0 the dynamic elements of the phenotype. 
	void preTrialReset();

	// Only used when CONTINUOUS LEARNING is not defined, in which case it updates wL with avgH.
	void postTrialUpdate();

	int inputSize, outputSize;

	std::unique_ptr<ComplexNode_G> topNodeG;

	std::vector<std::unique_ptr<ComplexNode_G>> complexGenome;

	// The phenotype is expected to be created and destroyed only once per Network, but it could happen several times.
	std::unique_ptr<ComplexNode_P> topNodeP;



	// Arrays for plasticity based updates. Contain all presynaptic and postSynaptic activities.
	// Must be : - reset to all 0s at the start of each trial;
	//			 - created alongside ComplexNode_P creation; 
	//           - freed alongside ComplexNode_P deletion.
	// Layout detailed in the Phenotype structs.
	std::unique_ptr<float[]> postSynActs, preSynActs;

#ifdef STDP
	// same size and layout that of preSynActs.
	std::unique_ptr<float[]> accumulatedPreSynActs;
#endif

	// size of postSynActs
	int postSynActArraySize;

	// size of preSynActs
	int preSynActsArraySize;

	// How many inferences were performed since last call to preTrialReset by the phenotype.
	int nInferencesOverTrial;

	// How many inferences were performed since phenotype creation.
	int nInferencesOverLifetime;

	// How many trials the phenotype has experimented.
	int nExperiencedTrials;

#ifdef SATURATION_PENALIZING
	// Sum over all the phenotype's activations, over the lifetime, of powf(activations[i], 2*n), n=typically 10.
	float saturationPenalization;

	// Follows the same usage pattern as the 4 arrays for plasticity updates. Size averageActivationArraySize.
	// Used to store, for each activation function of the phenotype, its average output over lifetime, for use in
	// getSaturationPenalization(). So set to 0 at phenotype creation, and never touched again.
	std::unique_ptr<float[]> averageActivation;

	// size of the averageActivation array.
	int averageActivationArraySize;

	float getSaturationPenalization();
#endif
};
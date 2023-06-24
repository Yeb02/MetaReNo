#pragma once

#include "Network.h"
#include <iostream>

Network::Network(Network* n) {
	inputSize = n->inputSize;
	outputSize = n->outputSize;

	// Complex genome
	complexGenome.resize(n->complexGenome.size());
	for (int i = 0; i < n->complexGenome.size(); i++) {
		complexGenome[i] = std::make_unique<ComplexNode_G>(n->complexGenome[i].get());
	}
	for (int i = 0; i < n->complexGenome.size(); i++) {
		// Setting pointers to children.
		// It can only be done once the genome has been constructed:

		complexGenome[i]->complexChildren.resize(n->complexGenome[i]->complexChildren.size());
		for (int j = 0; j < n->complexGenome[i]->complexChildren.size(); j++) {
			complexGenome[i]->complexChildren[j] = complexGenome[n->complexGenome[i]->complexChildren[j]->position].get();
		}
	}
	

	topNodeG = std::make_unique<ComplexNode_G>(n->topNodeG.get());

	// Setting pointers to children:
	topNodeG->complexChildren.resize(n->topNodeG->complexChildren.size());
	for (int j = 0; j < n->topNodeG->complexChildren.size(); j++) {
		topNodeG->complexChildren[j] = complexGenome[n->topNodeG->complexChildren[j]->position].get();
	}

	// The problem this solves is that we call mutate() on a newborn child and not on the parent that has finished
	// its lifetime (it would not make sense otherwise). 
	nInferencesOverLifetime = n->nInferencesOverLifetime;
	nExperiencedTrials = n->nExperiencedTrials;


	topNodeP.reset(NULL);
}


Network::Network(int inputSize, int outputSize) :
	inputSize(inputSize), outputSize(outputSize)
{

	complexGenome.resize(0);
	
	
	topNodeG = std::make_unique<ComplexNode_G>(inputSize, outputSize);
	topNodeG->position = (int)complexGenome.size();
	topNodeG->createInternalConnexions();

	
	topNodeP.reset(NULL);
}


float* Network::getOutput() {
	return topNodeP->preSynActs;
}


void Network::postTrialUpdate() {

	if (nInferencesOverTrial != 0) {
		
#ifndef CONTINUOUS_LEARNING
		topNodeP->updateWatTrialEnd(.1f / (float)nInferencesOverTrial); // the argument averages H.
#endif
	}
	else {
		std::cerr << "ERROR : postTrialUpdate WAS CALLED BEFORE EVALUATION ON A TRIAL !!" << std::endl;
	}

}


void Network::destroyPhenotype() {
	topNodeP.reset(NULL);
	postSynActs.reset(NULL);
	preSynActs.reset(NULL);
#ifdef SATURATION_PENALIZING
	averageActivation.reset(NULL);
#endif
#ifdef STDP
	accumulatedPreSynActs.reset(NULL);
#endif

}


void Network::createPhenotype() {
	if (topNodeP.get() == NULL) {
		topNodeP.reset(new ComplexNode_P(topNodeG.get()));

		std::vector<int> genomeState(complexGenome.size() + 1);

		topNodeG->computePreSynActArraySize(genomeState);
		preSynActsArraySize = genomeState[(int)complexGenome.size()];
		preSynActs = std::make_unique<float[]>(preSynActsArraySize);

		float* ptr_accumulatedPreSynActs = nullptr;
#ifdef STDP
		accumulatedPreSynActs = std::make_unique<float[]>(preSynActsArraySize);
		ptr_accumulatedPreSynActs = accumulatedPreSynActs.get();
#endif

		std::fill(genomeState.begin(), genomeState.end(), 0);

		topNodeG->computePostSynActArraySize(genomeState);
		postSynActArraySize = genomeState[(int)complexGenome.size()];
		postSynActs = std::make_unique<float[]>(postSynActArraySize);

		float* ptr_averageActivation = nullptr;
#ifdef SATURATION_PENALIZING
		std::fill(genomeState.begin(), genomeState.end(), 0);
		topNodeG->computeSaturationArraySize(genomeState);
		averageActivationArraySize = genomeState[(int)complexGenome.size()];
		averageActivation = std::make_unique<float[]>(averageActivationArraySize);
		ptr_averageActivation = averageActivation.get();

		saturationPenalization = 0.0f;
		topNodeP->setglobalSaturationAccumulator(&saturationPenalization);
		std::fill(averageActivation.get(), averageActivation.get() + averageActivationArraySize, 0.0f);
#endif

		
		// The following values will be modified by each node of the phenotype as the pointers are set.
		float* ptr_postSynActs = postSynActs.get();
		float* ptr_preSynActs = preSynActs.get();
		topNodeP->setArrayPointers(
			&ptr_postSynActs,
			&ptr_preSynActs,
			&ptr_averageActivation,
			&ptr_accumulatedPreSynActs
		);

		nInferencesOverTrial = 0;
		nInferencesOverLifetime = 0;
		nExperiencedTrials = 0;
	}
};


void Network::preTrialReset() {
	nInferencesOverTrial = 0;
	nExperiencedTrials++;
	std::fill(postSynActs.get(), postSynActs.get() + postSynActArraySize, 0.0f);
	//std::fill(preSynActs.get(), preSynActs.get() + preSynActsArraySize, 0.0f); // is already set to the biases.
#ifdef STDP
	std::fill(accumulatedPreSynActs.get(), accumulatedPreSynActs.get() + preSynActsArraySize, 0.0f);
#endif
	

	topNodeP->preTrialReset();
};


void Network::step(const std::vector<float>& obs) {
	nInferencesOverLifetime++;
	nInferencesOverTrial++;

	std::copy(obs.begin(), obs.end(), topNodeP->postSynActs);
	for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
		topNodeP->totalM[i] = 0.0f;
	}
	topNodeP->forward();
}


#ifdef SATURATION_PENALIZING
float Network::getSaturationPenalization()
{
	if (nInferencesOverLifetime == 0) {
		std::cerr <<
			"ERROR : getSaturationPenalization() WAS CALLED, BUT THE PHENOTYPE HAS NEVER BEEN USED BEFORE !"
			<< std::endl;
		return 0.0f;
	}

	
	float p1 = averageActivationArraySize != 0 ? saturationPenalization / (nInferencesOverLifetime * averageActivationArraySize) : 0.0f;


	float p2 = 0.0f;
	float invNInferencesN = 1.0f / nInferencesOverLifetime;
	for (int i = 0; i < averageActivationArraySize; i++) {
		p2 += powf(abs(averageActivation[i]) * invNInferencesN, 6.0f);
	}
	p2 /= (float) averageActivationArraySize;
	


	constexpr float µ = .5f;
	return µ * p1 + (1 - µ) * p2;
}
#endif


void Network::save(std::ofstream& os)
{
	int version = 0;
	WRITE_4B(version, os); // version

	WRITE_4B(inputSize, os);
	WRITE_4B(outputSize, os);

	topNodeG->save(os);

	int _s = (int)complexGenome.size();
	WRITE_4B(_s, os);
	for (int i = 0; i < complexGenome.size(); i++) {
		complexGenome[i]->save(os);
	}
	

	for (int i = 0; i < topNodeG->complexChildren.size(); i++) {
		WRITE_4B(topNodeG->complexChildren[i]->position, os);
	}
	

	for (int i = 0; i < complexGenome.size(); i++) {
		for (int j = 0; j < complexGenome[i]->complexChildren.size(); j++) {
			WRITE_4B(complexGenome[i]->complexChildren[j]->position, os);
		}
	}

}

Network::Network(std::ifstream& is)
{
	int version;
	READ_4B(version, is);
	
	READ_4B(inputSize, is);
	READ_4B(outputSize, is);

	topNodeG = std::make_unique<ComplexNode_G>(is);

	int _s;
	READ_4B(_s, is);
	complexGenome.resize(_s);
	for (int i = 0; i < complexGenome.size(); i++) {
		complexGenome[i] = std::make_unique<ComplexNode_G>(is);
		complexGenome[i]->position = i;
	}
	topNodeG->position = (int)complexGenome.size();


	for (int i = 0; i < topNodeG->complexChildren.size(); i++) {
		READ_4B(_s, is);
		topNodeG->complexChildren[i] = complexGenome[_s].get();
	}
	
	for (int i = 0; i < complexGenome.size(); i++) {
		for (int j = 0; j < complexGenome[i]->complexChildren.size(); j++) {
			READ_4B(_s, is);
			complexGenome[i]->complexChildren[j] = complexGenome[_s].get();
		}
	}

	topNodeP.reset(NULL);
}
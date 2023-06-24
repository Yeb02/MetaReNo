#pragma once


#include "TorchNNs.h"
#include "Trial.h"
#include "Network.h"
#include "config.h"

struct GeneratorParameters {

	float lr;
	int nNetsPerBatch;
	int nTrialsPerNet;
	int gaussianVecSize;
	float elitePercentage;
	int nUpdatedPoints;
	float updateRadius;

	//defaults:
	GeneratorParameters()
	{
		lr = .1f;
		nNetsPerBatch = 30;
		nTrialsPerNet = 20;
		gaussianVecSize = 10;
		elitePercentage = .2f;
		nUpdatedPoints = 3*nNetsPerBatch;
		updateRadius = .2f;
	}
};



struct Generator 
{
	Generator(int netInSize, int netOutSize);
	~Generator() {};

	void setParameters(GeneratorParameters& params) 
	{
		lr = params.lr;
		nNetsPerBatch = params.nNetsPerBatch;
		nTrialsPerNet = params.nTrialsPerNet;
		gaussianVecSize = params.gaussianVecSize;
		elitePercentage = params.elitePercentage;
		nUpdatedPoints = params.nUpdatedPoints;
		updateRadius = params.updateRadius;
	};

	void step(Trial* trial);

	// Does not store grads.
	Network* createNet(float* seed, std::vector<torch::Tensor>& rawParameters);

	// Grads are not accumulated at network creation, because we do not know their respective 
	// coefficients yet. They are not stored either, as it would take way too much memory to be
	// practical. The best solution is to call forward a second time on the same inputs after network
	// evaluation.
	void accumulateGrads(float* updatedSeed, std::vector<torch::Tensor>& intendedOutputs);

	void save();
	void save1Net();

	

	// TODO an array of those.
	int nNodes;
	std::unique_ptr<Matrixator> matrixator;
	std::unique_ptr<Specialist> matSpecialists[N_MATRICES];
	std::unique_ptr<Specialist> arrSpecialists[N_ARRAYS];


	std::unique_ptr<torch::optim::SGD> optimizer;


	std::vector<int> nLines; 
	std::vector<int> nCols;

	std::vector<int> colEmbS;
	std::vector<int> lineEmbS;


	int netInSize, netOutSize;

	
	float lr;
	int gaussianVecSize;
	int nNetsPerBatch;
	int nTrialsPerNet;
	float elitePercentage;
	int nUpdatedPoints;
	float updateRadius;
};

#include "ComplexNode_P.h"



ComplexNode_P::ComplexNode_P(ComplexNode_G* _type) : 
	type(_type),
	toComplex(&_type->toComplex),
	toModulation(&_type->toModulation),
	toOutput(&_type->toOutput)
{
	// create COMPLEX children recursively 
	complexChildren.reserve(type->complexChildren.size());
	for (int i = 0; i < type->complexChildren.size(); i++) {
		complexChildren.emplace_back(type->complexChildren[i]);
	}

	// TotalM is not initialized (i.e. zeroed) here because a call to preTrialReset() 
	// must be made before any forward pass. 
};


#ifndef CONTINUOUS_LEARNING
void ComplexNode_P::updateWatTrialEnd(float invNInferencesOverTrial) {

	// return; This completely disables wL (when CONTINUOUS_LEARNING is disabled)

	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].updateWatTrialEnd(invNInferencesOverTrial);
	}

	toComplex.updateWatTrialEnd(invNInferencesOverTrial);
	toMemory.updateWatTrialEnd(invNInferencesOverTrial);
	toModulation.updateWatTrialEnd(invNInferencesOverTrial);
	toOutput.updateWatTrialEnd(invNInferencesOverTrial);
	
	for (int i = 0; i < memoryChildren.size(); i++) {
		memoryChildren[i].updateWatTrialEnd(invNInferencesOverTrial);
	}
}
#endif

void ComplexNode_P::setArrayPointers(float** post_syn_acts, float** pre_syn_acts, float** aa, float** acc_pre_syn_acts) {

	// TODO ? if the program runs out of heap memory, one could make it so that a node does not store its own 
	// output. But prevents in place matmul, and complexifies things.

	postSynActs = *post_syn_acts;
	preSynActs = *pre_syn_acts;


	*post_syn_acts += type->inputSize + MODULATION_VECTOR_SIZE;
	*pre_syn_acts += type->outputSize + MODULATION_VECTOR_SIZE;


#ifdef SATURATION_PENALIZING
	averageActivation = *aa;
	*aa += MODULATION_VECTOR_SIZE;
#endif

#ifdef STDP
	accumulatedPreSynActs = *acc_pre_syn_acts;
	*acc_pre_syn_acts += type->outputSize + MODULATION_VECTOR_SIZE;
#endif

	for (int i = 0; i < complexChildren.size(); i++) {
		*post_syn_acts += complexChildren[i].type->outputSize;
		*pre_syn_acts += complexChildren[i].type->inputSize;
#ifdef STDP
		*acc_pre_syn_acts += complexChildren[i].type->inputSize;
#endif
#ifdef SATURATION_PENALIZING
		*aa += complexChildren[i].type->inputSize;
#endif
	}


	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].setArrayPointers(post_syn_acts, pre_syn_acts, aa, acc_pre_syn_acts);
	}
}


void ComplexNode_P::preTrialReset() {

	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].preTrialReset();
	}


	toComplex.zero();
	toModulation.zero();
	toOutput.zero();


#ifdef RANDOM_WB
	toComplex.randomInitWB();
	toModulation.randomInitWB();
	toOutput.randomInitWB();
#endif

#if defined(CONTINUOUS_LEARNING) && defined(ZERO_WL_BEFORE_TRIAL)
	toComplex.zeroWlifetime();
	toModulation.zeroWlifetime();
	toOutput.zeroWlifetime();
#endif 
	
}


#ifdef SATURATION_PENALIZING
void ComplexNode_P::setglobalSaturationAccumulator(float* globalSaturationAccumulator) {
	this->globalSaturationAccumulator = globalSaturationAccumulator;
	for (int i = 0; i < complexChildren.size(); i++) {
		complexChildren[i].setglobalSaturationAccumulator(globalSaturationAccumulator);
	}
}
#endif


void ComplexNode_P::forward() {

#ifdef SATURATION_PENALIZING
	constexpr float saturationExponent = 6.0f; 
#endif

	

#ifdef DROPOUT
	toComplex.dropout();
	toOutput.dropout();
	toModulation.dropout();
#endif

	// STEP 1 to 4: propagate and call children's forward.
	// 1_Modulation -> 2_Complex -> 3_Modulation -> 4_output
	// This could be done simultaneously for all types, but doing it this way drastically speeds up information transmission
	// through the network. 


	// These 3 lambdas, hopefully inline, avoid repetition, as they are used for each child type.

	auto propagate = [this](InternalConnexion_P& icp, float* destinationArray)
	{
		int nl = icp.type->nLines;
		int nc = icp.type->nColumns;
		int matID = 0;

		float* H = icp.H.get();
		float* wLifetime = icp.wLifetime.get();
		float* alpha = icp.type->alpha.get();

#ifdef RANDOM_WB
		float* w = icp.w.get();
		float* b = icp.biases.get();
#else
		float* w = icp.type->w.get();
		float* b = icp.type->biases.get();
#endif
		

			
		for (int i = 0; i < nl; i++) {
			destinationArray[i] = b[i];
			for (int j = 0; j < nc; j++) {
				// += (H * alpha + w + wL) * prevAct
				destinationArray[i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * postSynActs[j];
				matID++;
			}
		}


	};

	auto hebbianUpdate = [this](InternalConnexion_P& icp, float* destinationArray) {
		int nl = icp.type->nLines;
		int nc = icp.type->nColumns;
		int matID = 0;

		float* A = icp.type->A.get();
		float* B = icp.type->B.get();
		float* C = icp.type->C.get();
		float* D = icp.type->D.get();
		float* eta = icp.type->eta.get();
		float* H = icp.H.get();
		float* E = icp.E.get();

#ifdef CONTINUOUS_LEARNING
		float* wLifetime = icp.wLifetime.get();
		float* gamma = icp.type->gamma.get();
		float* alpha = icp.type->alpha.get();
#else
		float* avgH = icp.avgH.get();
#endif


#ifdef OJA
		float* delta = icp.type->delta.get();
#ifdef RANDOM_WB
		float* w = icp.w.get();
#else
		float* w = icp.type->w.get();
#endif
#ifndef CONTINUOUS_LEARNING
		float* wLifetime = icp.wLifetime.get();
		float* alpha = icp.type->alpha.get();
#endif
#endif


		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {
#ifdef CONTINUOUS_LEARNING
				wLifetime[matID] = (1 - gamma[matID]) * wLifetime[matID] + gamma[matID] * H[matID] * alpha[matID] * totalM[1]; // TODO remove ?
#endif
				E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
					(A[matID] * destinationArray[i] * postSynActs[j] + B[matID] * destinationArray[i] + C[matID] * postSynActs[j] + D[matID]);

#ifdef OJA
				E[matID] -= eta[matID] * destinationArray[i] * destinationArray[i] * delta[matID] * (w[matID] + alpha[matID]*H[matID] + wLifetime[matID]);
#endif

				H[matID] += E[matID] * totalM[0];
				H[matID] = std::clamp(H[matID], -1.0f, 1.0f);
#ifndef CONTINUOUS_LEARNING
				avgH[matID] += H[matID];
#endif
				matID++;

			}
		}
	};

	auto applyNonLinearities = [](float* src, float* dst, int size
#ifdef STDP
		, float* acc_src, float* mu, float* lambda
#endif
		) 
	{
#ifdef STDP
		for (int i = 0; i < size; i++) {
			acc_src[i] = acc_src[i] * (1.0f-mu[i]) + src[i]; // * mu[i] ? TODO
		}

		src = acc_src;
#endif
		
		for (int i = 0; i < size; i++) {
			dst[i] = tanhf(src[i]);
		
			if (src[i] != src[i] || dst[i] != dst[i]) {
				__debugbreak();
			}
		}

#ifdef STDP
		for (int i = 0; i < size; i++) {
			acc_src[i] -= lambda[i] * (1.0f - dst[i] * dst[i]) * powf(dst[i], 2.0f * 0.0f + 1.0f); // TODO only works for tanh as of now
		}
#endif
	};



	// STEP 1: MODULATION  A
	{
		propagate(toModulation, preSynActs + type->outputSize);
		applyNonLinearities(
			preSynActs + type->outputSize,
			postSynActs + type->inputSize, 
			MODULATION_VECTOR_SIZE
#ifdef STDP
			, accumulatedPreSynActs + type->outputSize, type->toModulation.STDP_mu.get(), type->toModulation.STDP_lambda.get()
#endif
		);
		hebbianUpdate(toModulation, postSynActs + type->inputSize);

		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			totalM[i] += postSynActs[i + type->inputSize];
		}

#ifdef SATURATION_PENALIZING 
		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			float v = postSynActs[i + type->inputSize];
			*globalSaturationAccumulator += powf(abs(v), saturationExponent);
			averageActivation[i] += v;
		}
#endif
	}


	// STEP 2: COMPLEX
	if (complexChildren.size() != 0) {
		float* ptrToInputs = preSynActs + type->outputSize + MODULATION_VECTOR_SIZE;
#ifdef STDP
		float* ptrToAccInputs = accumulatedPreSynActs + type->outputSize + MODULATION_VECTOR_SIZE;
#endif
		propagate(toComplex, ptrToInputs);
		
		

		// Apply non-linearities
		int id = 0;
		for (int i = 0; i < complexChildren.size(); i++) {


			applyNonLinearities(
				ptrToInputs + id,
				complexChildren[i].postSynActs,
				complexChildren[i].type->inputSize
#ifdef STDP
				, ptrToAccInputs + id, &type->toComplex.STDP_mu[id], &type->toComplex.STDP_lambda[id]
#endif
			);

#ifdef SATURATION_PENALIZING 
			// child post-syn input
			int i0 = MODULATION_VECTOR_SIZE + id;
			for (int j = 0; j < complexChildren[i].type->inputSize; j++) {
				float v = complexChildren[i].postSynActs[j];
				*globalSaturationAccumulator += powf(abs(v), saturationExponent);
				averageActivation[i0+j] += v;
			}
#endif

			id += complexChildren[i].type->inputSize;
		}

		// has to happen after non linearities but before forward, 
		// for children's output not to have changed yet.
		hebbianUpdate(toComplex, ptrToInputs);


		// transmit modulation and apply forward, then retrieve the child's output.

		float* childOut = postSynActs + type->inputSize + MODULATION_VECTOR_SIZE;
		for (int i = 0; i < complexChildren.size(); i++) {

			for (int j = 0; j < MODULATION_VECTOR_SIZE; j++) {
				complexChildren[i].totalM[j] = this->totalM[j];
			}

			complexChildren[i].forward();

			std::copy(complexChildren[i].preSynActs, complexChildren[i].preSynActs + complexChildren[i].type->outputSize, childOut);
			childOut += complexChildren[i].type->outputSize;
		}

	}


	// STEP 3: MODULATION B. 
	if (complexChildren.size() != 0)
	{
		propagate(toModulation, preSynActs + type->outputSize);
		applyNonLinearities(
			preSynActs + type->outputSize,
			postSynActs + type->inputSize,
			MODULATION_VECTOR_SIZE
#ifdef STDP
			, accumulatedPreSynActs + type->outputSize, type->toModulation.STDP_mu.get(), type->toModulation.STDP_lambda.get()
#endif
		);
		hebbianUpdate(toModulation, postSynActs + type->inputSize);

		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			totalM[i] += postSynActs[i + type->inputSize];
		}

#ifdef SATURATION_PENALIZING 
		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			float v = postSynActs[i + type->inputSize];
			*globalSaturationAccumulator += powf(abs(v), saturationExponent);
			averageActivation[i] += v;
		}
#endif
	}


	// STEP 6: OUTPUT
	{
		propagate(toOutput, preSynActs);
		
		
		applyNonLinearities(
			preSynActs,
			preSynActs,
			type->outputSize
#ifdef STDP
			, accumulatedPreSynActs, type->toOutput.STDP_mu.get(), type->toOutput.STDP_lambda.get()
#endif
		);

		hebbianUpdate(toOutput, preSynActs);
	}
	
}

#pragma once

#include "Generator.h"

// src is unchanged.
void normalizeArray(float* src, float* dst, int size) {
	float avg = 0.0f;
	for (int i = 0; i < size; i++) {
		avg += src[i];
	}
	avg /= (float)size;
	float variance = 0.0f;
	for (int i = 0; i < size; i++) {
		dst[i] = src[i] - avg;
		variance += dst[i] * dst[i];
	}
	if (variance < .001f) return;
	float InvStddev = 1.0f / sqrtf(variance / (float)size);
	for (int i = 0; i < size; i++) {
		dst[i] *= InvStddev;
	}
}

// src is unchanged. 
void rankArray(float* src, std::vector<int>& dst, int size) {
	for (int i = 0; i < size; i++) {
		dst[i] = i;
	}
	// sort dst in decreasing order.
	std::sort(dst.begin(), dst.end(), [src](int a, int b) -> bool
		{
			return src[a] > src[b];
		}
	);

	return;
}



Generator::Generator(int netInSize, int netOutSize) :
	netInSize(netInSize), netOutSize(netOutSize)
{ 

	GeneratorParameters params; 
	setParameters(params); 

	std::vector<at::Tensor> optimizerParams;

	nNodes = 1;

	nLines.push_back(netOutSize + MODULATION_VECTOR_SIZE) ;
	nCols.push_back(netInSize + MODULATION_VECTOR_SIZE) ;
	
	for (int ni = 0; ni < nNodes; ni++)
	{
		colEmbS.push_back(3 + (int)sqrtf((float)nCols[ni])); // nCols and not nLines, because what matters is how many there are and not their sizes.
		lineEmbS.push_back(3 + (int)sqrtf((float)nLines[ni])); // same.

		matrixator = std::make_unique<Matrixator>(gaussianVecSize, nCols[ni], nLines[ni], colEmbS[ni], lineEmbS[ni]);

		int in, out, i;

		i = 0;
		in = colEmbS[ni] * nCols[ni] + lineEmbS[ni];
		out = nCols[ni];

		// in this order : A,B, C, D, eta, alpha, gamma?, w?, delta?.
		for (int i = 0; i < N_MATRICES; i++) 
		{
			matSpecialists[i] = std::make_unique<Specialist>(in, out);
			std::vector<at::Tensor> params = matSpecialists[i]->parameters();
			optimizerParams.insert(optimizerParams.end(), params.begin(), params.end());
		}


		i = 0;
		in = lineEmbS[ni] * nLines[ni];
		out = nLines[ni];
		// in this order: biases?, mu?, lambda?
		for (int i = 0; i < N_ARRAYS; i++)
		{
			arrSpecialists[i] = std::make_unique<Specialist>(in, out);
			std::vector<at::Tensor> params = arrSpecialists[i]->parameters();
			optimizerParams.insert(optimizerParams.end(), params.begin(), params.end());
		}

	}

	optimizer = std::make_unique<torch::optim::SGD>(optimizerParams, lr);
}



void Generator::step(Trial* trial) 
{
	std::vector<float> rawScores;
	std::vector<int> ranks;
	rawScores.resize(nNetsPerBatch);
	ranks.resize(nNetsPerBatch);

	// Holds all the meta-NNs outputs for each generated network of the batch.
	std::vector<std::vector<torch::Tensor>> NNOutputs;
	NNOutputs.resize(nNetsPerBatch);

	// Create seeds. Each seed is normalized. TODO maybe it is not a good idea
	std::unique_ptr<float[]> seeds = std::make_unique<float[]>(gaussianVecSize * nNetsPerBatch);
	for (int i = 0; i < nNetsPerBatch; i++) {
		float s = 0.0f;
		for (int j = 0; j < gaussianVecSize; j++) {
			seeds[i * gaussianVecSize + j] = NORMAL_01;
			s += seeds[i * gaussianVecSize + j] * seeds[i * gaussianVecSize + j];
		}
		s = powf(s, -.5f);
		for (int j = 0; j < gaussianVecSize; j++) {
			seeds[i * gaussianVecSize + j] *= s;
		}
	}


	// Create and evaluate each model from the seeds.
	float avgS = 0.0f, maxS = -1000.0f;
	for (int i = 0; i < nNetsPerBatch; i++) {
		Network* n = createNet(&seeds[gaussianVecSize * i], NNOutputs[i]);
		n->createPhenotype();

		rawScores[i] = 0.0f;

		for (int j = 0; j < nTrialsPerNet; j++) {

			n->preTrialReset();
			trial->reset(false);

			while (!trial->isTrialOver) {
				n->step(trial->observations);
				trial->step(n->getOutput());
			}

			n->postTrialUpdate();
			rawScores[i] += trial->score;
		}

		//rawScores[i] = n->topNodeG->toModulation.alpha[0];

		if (rawScores[i] > maxS) maxS = rawScores[i];
		avgS += rawScores[i];

		delete n;
		
	}

	maxS /= (float)nTrialsPerNet;
	//avgS /= (float)nNetsPerBatch;
	avgS /= (float)(nTrialsPerNet * nNetsPerBatch);
	std::cout << "Max : " << maxS << " , avg : " << avgS << std::endl;
	rankArray(rawScores.data(), ranks, nNetsPerBatch);

	// Apply gradients in points neighboring the seeds.
	int nUpdatesPerEliteSeed = nUpdatedPoints / (int)((float)nNetsPerBatch * elitePercentage);
	float* updatedSeed = new float[gaussianVecSize];
	for (int i = 0; i < (int) ((float)nNetsPerBatch * elitePercentage); i++) {

		float* s0 = &seeds[ranks[i]];

		for (int j = 0; j < nUpdatesPerEliteSeed; j++) {

			float s = 0.0f;
			for (int j = 0; j < gaussianVecSize; j++) {
				updatedSeed[j] = s0[j] + NORMAL_01 * updateRadius;
				s += updatedSeed[j] * updatedSeed[j];
			}
			s = powf(s, -.5f);
			for (int j = 0; j < gaussianVecSize; j++) {
				updatedSeed[j] *= s;
			}

			accumulateGrads(&seeds[gaussianVecSize * i], NNOutputs[ranks[i]]);
		}
	}
	
	/*torch::optim::OptimizerOptions options;
	options.set_lr(lr);
	optimizer->param_groups()[0].set_options(std::make_unique<torch::optim::OptimizerOptions>(options));*/ // TODO fishy

	optimizer->step();
	optimizer->zero_grad(); 

	delete[] updatedSeed;
}

// both tensor.index and tensor.cat let gradients flow through (if the initial tensor had
// requires_grad set to true).
void Generator::accumulateGrads(float* updatedSeed, std::vector<torch::Tensor>& intendedOutputs)
{
	torch::Tensor gaussianInputVector = torch::from_blob(updatedSeed, { 1, gaussianVecSize });

	int intendedOutputsID = 0;
	
	for (int _ni = 0; _ni < nNodes; _ni++) {

		// colEmb0, colEmb1, ... colEmb(nCols-1), lineEmb0, ... lineEmb(nLines-1).
		torch::Tensor linesAndColsEmbeddings = matrixator->forward(gaussianInputVector.reshape({1, gaussianVecSize}));
		

		// MATRICES :
		{
			torch::Tensor external_grad = torch::ones({ nLines[_ni], nCols[_ni]}, torch::dtype(torch::kFloat32));

			torch::Tensor colEmbeddings1D = linesAndColsEmbeddings.index({ 0, torch::indexing::Slice(0, colEmbS[_ni] * nCols[_ni]) });
			torch::Tensor colEmbeddings = colEmbeddings1D.repeat({ nLines[_ni], 1 });
			

			torch::Tensor lineEmbeddings = torch::zeros({ nLines[_ni], lineEmbS[_ni] }, torch::dtype(torch::kFloat32));
			for (int i = 0; i < nLines[_ni]; i++) {
				int i0 = colEmbS[_ni] * nCols[_ni] + i * lineEmbS[_ni];
				for (int j = 0; j < lineEmbS[_ni]; j++) {
					lineEmbeddings.index_put_({ i,j }, linesAndColsEmbeddings.index({ 0, i0 + j }));
				}
			}
			torch::Tensor specialistInput = torch::cat({ colEmbeddings, lineEmbeddings }, 1);

			auto callForward = [&](Specialist* s)
			{
				torch::Tensor matSpecialistOutput = s->forward(specialistInput);
				torch::Tensor loss = torch::mse_loss(matSpecialistOutput, intendedOutputs[intendedOutputsID++]);
				loss.backward({}, true);
			};

			int matSpeID = 0;
			callForward(matSpecialists[matSpeID++].get()); // A
			callForward(matSpecialists[matSpeID++].get()); // B
			callForward(matSpecialists[matSpeID++].get()); // C
			callForward(matSpecialists[matSpeID++].get()); // D
			callForward(matSpecialists[matSpeID++].get()); // eta
			callForward(matSpecialists[matSpeID++].get()); // alpha
#ifdef CONTINUOUS_LEARNING
			callForward(matSpecialists[matSpeID++].get()); // gamma
#endif
#ifndef RANDOM_WB
			callForward(matSpecialists[matSpeID++].get()); // w
#endif 
#ifdef OJA
			callForward(matSpecialists[matSpeID++].get()); // delta	
#endif
			
		}


		// ARRAYS :
		if (N_ARRAYS > 0) {
			int iFirst = nCols[_ni] * colEmbS[_ni];
			int iLast = nCols[_ni] * colEmbS[_ni] + nLines[_ni] * lineEmbS[_ni];
			torch::Tensor specialistInput = linesAndColsEmbeddings.index({ "...", torch::indexing::Slice(iFirst, iLast)});

			torch::Tensor external_grad = torch::ones({ 1,nLines[_ni] }, torch::dtype(torch::kFloat32)) ;

			auto callForward = [&](Specialist* s)
			{
				torch::Tensor arrSpecialistOutput = s->forward(specialistInput);
				torch::Tensor loss = torch::mse_loss(arrSpecialistOutput, intendedOutputs[intendedOutputsID++]);
				loss.backward({}, true);
			};

			int matSpeID = 0;

#ifndef RANDOM_WB
			callForward(arrSpecialists[matSpeID++].get()); // biases
#endif 

#ifdef STDP
			callForward(arrSpecialists[matSpeID++].get()); // mu
			callForward(arrSpecialists[matSpeID++].get()); // lambda
#endif 

		}
	}
}

Network* Generator::createNet(float * seed, std::vector<torch::Tensor>& rawParameters)
{
	torch::NoGradGuard no_grad;

	Network* n = new Network(netInSize, netOutSize);

	torch::Tensor gaussianInputVector = torch::from_blob(seed, { 1, gaussianVecSize });

	rawParameters.reserve(((int)n->complexGenome.size() + 1) * (N_MATRICES + N_ARRAYS));

	for (int _ni = 0; _ni < n->complexGenome.size()+1; _ni++) {
		ComplexNode_G* node = _ni == n->complexGenome.size() ? n->topNodeG.get() : n->complexGenome[_ni].get();

		// colEmb0, colEmb1, ... colEmb(nCols-1), lineEmb0, ... lineEmb(nLines-1).
		torch::Tensor linesAndColsEmbeddings = matrixator->forward(gaussianInputVector);

		// MATRICES :
		{
			torch::Tensor colEmbeddings1D = linesAndColsEmbeddings.index({ 0, torch::indexing::Slice(0, colEmbS[_ni] * nCols[_ni]) });
			torch::Tensor colEmbeddings = colEmbeddings1D.repeat({ nLines[_ni], 1 });

			torch::Tensor lineEmbeddings = torch::zeros({ nLines[_ni], lineEmbS[_ni] }, torch::dtype(torch::kFloat32));
			for (int i = 0; i < nLines[_ni]; i++) {
				int i0 = colEmbS[_ni] * nCols[_ni] + i * lineEmbS[_ni];
				for (int j = 0; j < lineEmbS[_ni]; j++) {
					lineEmbeddings.index_put_({ i,j }, linesAndColsEmbeddings.index({ 0, i0 + j }));
				}
			}
			torch::Tensor matSpecialistInput = torch::cat({ colEmbeddings, lineEmbeddings }, 1);
			

			// This block is horrendous but I dont see any way around it without rearchitecturing
			auto fillMat = [this, &matSpecialistInput, &_ni, &rawParameters](int speID, ComplexNode_G* node)
			{

				torch::Tensor matSpecialistOutput = matSpecialists[speID]->forward(matSpecialistInput);

				rawParameters.emplace_back(matSpecialistOutput);

				auto accessor = matSpecialistOutput.accessor<float, 2>(); 

				auto fillLines = [&accessor, this, speID, _ni](int line0, InternalConnexion_G* co) {
					float* mat = nullptr;
					bool is01;

					// Switch does not accomodate conditional compilation easily.
					{
						int i = 0;
						if (i++ == speID) {
							mat = co->A.get(); is01 = false;
						} 
						else if (i++ == speID) {
							mat = co->B.get(); is01 = false;
						}
						else if (i++ == speID) {
							mat = co->C.get(); is01 = false;
						}
						else if (i++ == speID) {
							mat = co->D.get(); is01 = false;
						}
						else if(i++ == speID) {
							mat = co->eta.get();  is01 = true;
						}
						else if (i++ == speID) {
							mat = co->alpha.get(); is01 = false;
						}
#ifdef CONTINUOUS_LEARNING
						else if (i++ == speID) {
							mat = co->gamma.get();  is01 = true;
						}
#endif 

#ifndef RANDOM_WB
						else if (i++ == speID) {
							mat = co->w.get(); is01 = false;
						}
#endif 

#ifdef OJA
						else if (i++ == speID) {
							mat = co->delta.get();  is01 = true;
						}
#endif
					}


					int matID = 0;
					if (is01) {
						for (int i = line0; i < line0 + co->nLines; i++) {
							for (int j = 0; j < nCols[_ni]; j++) {
								float invTau = powf(2.0f, -(5.0f * accessor[i][j] + 2.0f));
								mat[matID] = 1.0f - powf(2.0f, -invTau);
								matID++;
							}
						}
					}
					else {
						for (int i = line0; i < line0 + co->nLines; i++) {
							for (int j = 0; j < nCols[_ni]; j++) {
								mat[matID] = 2.0f * accessor[i][j];
								matID++;
							}
						}
					}
				};

				int l0 = 0;
				fillLines(l0, &node->toOutput);
				l0 += node->outputSize;
				fillLines(l0, &node->toModulation);
				l0 += MODULATION_VECTOR_SIZE;
				fillLines(l0, &node->toComplex);
			};

			int matSpeID = 0;
			fillMat(matSpeID++, node); // A
			fillMat(matSpeID++, node); // B
			fillMat(matSpeID++, node); // C
			fillMat(matSpeID++, node); // D
			fillMat(matSpeID++, node); // eta
			fillMat(matSpeID++, node); // alpha
#ifdef CONTINUOUS_LEARNING
			fillMat(matSpeID++, node); // gamma
#endif
#ifndef RANDOM_WB
			fillMat(matSpeID++, node); // w
#endif 
#ifdef OJA
			fillMat(matSpeID++, node); // delta	
#endif 
		}


		// ARRAYS :
		if (N_ARRAYS > 0) 
		{ 
			int id0 = colEmbS[_ni] * nCols[_ni];
			torch::Tensor arrSpecialistInput = linesAndColsEmbeddings.index({ "...", torch::indexing::Slice(id0, id0 + nLines[_ni] * lineEmbS[_ni])});

			auto fillArr = [this, &arrSpecialistInput, &rawParameters](int speID, ComplexNode_G* node)
			{

				torch::Tensor arrSpecialistOutput = arrSpecialists[speID]->forward(arrSpecialistInput);
				rawParameters.emplace_back(arrSpecialistOutput);

				auto accessor = arrSpecialistOutput.accessor<float, 2>(); 

				auto fillSubArr = [&accessor, this, speID](int i0, InternalConnexion_G* co) {
					float* mat = nullptr;
					bool is01 = false;

					
					int i = 0;

#ifndef RANDOM_WB
					if (i++ == speID) {
						mat = co->biases.get(); is01 = false;
					}
#endif

#ifdef STDP
					if (i++ == speID) {
						mat = co->STDP_mu.get();  is01 = true;
					}
					else if (i++ == speID) {
						mat = co->STDP_lambda.get();  is01 = true;
					}
#endif 
					if (is01) {
						for (int j = i0; j < i0 + co->nLines; j++) {
							float invTau = powf(2.0f, -(5.0f * accessor[0][j] + 2.0f));
							mat[j - i0] = 1.0f - powf(2.0f, -invTau);
						}
					} else {
						for (int j = i0; j < i0 + co->nLines; j++) {
							mat[j - i0] = 3.0f * accessor[0][j];
						}
					}
					
				};

				int i0 = 0;
				fillSubArr(i0, &node->toOutput);
				i0 += node->outputSize;
				fillSubArr(i0, &node->toModulation);
				i0 += MODULATION_VECTOR_SIZE;
				fillSubArr(i0, &node->toComplex);
			};

			int arrSpeID = 0;

#ifndef RANDOM_WB
			fillArr(arrSpeID++, node); // biases
#endif
			
#ifdef STDP
			fillArr(arrSpeID++, node); // mu
			fillArr(arrSpeID++, node); // lambda
#endif 

		}
	}


	return n;
}

void Generator::save() 
{

}

void Generator::save1Net() 
{

}

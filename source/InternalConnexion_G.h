#pragma once

#include <memory>
#include <fstream>

#define WRITE_4B(i, os) os.write(reinterpret_cast<const char*>(&i), 4);
#define READ_4B(i, is) is.read(reinterpret_cast<char*>(&i), 4);

#include "Random.h"
#include "config.h"


struct InternalConnexion_G {


	int nLines, nColumns;

	std::unique_ptr<float[]> A;
	std::unique_ptr<float[]> B;
	std::unique_ptr<float[]> C;
	std::unique_ptr<float[]> D;
	std::unique_ptr<float[]> eta;	// in [0, 1]
	std::unique_ptr<float[]> alpha;

	std::unique_ptr<ACTIVATION[]> activationFunctions;
	std::unique_ptr<float[]> biases;

#ifndef RANDOM_W
	std::unique_ptr<float[]> w;
#endif

#ifdef OJA
	std::unique_ptr<float[]> delta; // in [0, 1]
#endif

#ifdef CONTINUOUS_LEARNING
	std::unique_ptr<float[]> gamma; // in [0, 1]
#endif

#ifdef GUIDED_MUTATIONS
	std::unique_ptr<float[]> accumulator;

	void zeroAccumulator() {
		for (int i = 0; i < nLines * nColumns; i++) {
			accumulator[i] = 0.0f;
		}
	}
#endif


#ifdef STDP
	std::unique_ptr<float[]> STDP_mu;
	std::unique_ptr<float[]> STDP_lambda;
#endif


	InternalConnexion_G() { __debugbreak(); };

	InternalConnexion_G(int nLines, int nColumns);

	InternalConnexion_G(const InternalConnexion_G& gc);

	InternalConnexion_G operator=(const InternalConnexion_G& gc);

	~InternalConnexion_G() {};

	InternalConnexion_G(std::ifstream& is);
	void save(std::ofstream& os);

};

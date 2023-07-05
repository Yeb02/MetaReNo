#pragma once

#include "InternalConnexion_G.h"


InternalConnexion_G::InternalConnexion_G(int nLines, int nColumns) :
	nLines(nLines), nColumns(nColumns)
{

	int s = nLines * nColumns;
	 

	eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);
	D = std::make_unique<float[]>(s);
	alpha = std::make_unique<float[]>(s);

#ifndef RANDOM_WB
	w = std::make_unique<float[]>(s);
#endif

#ifdef OJA
	delta = std::make_unique<float[]>(s);
#endif
	
#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
#endif
	

	s = nLines;
#ifndef RANDOM_WB
	biases = std::make_unique<float[]>(s);
#endif
	

#ifdef STDP
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
#endif
}

InternalConnexion_G::InternalConnexion_G(const InternalConnexion_G& gc) {

	
	nLines = gc.nLines;
	nColumns = gc.nColumns;

	int s = nLines * nColumns;

	eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);
	D = std::make_unique<float[]>(s);
	alpha = std::make_unique<float[]>(s);
	

	std::copy(gc.eta.get(), gc.eta.get() + s, eta.get());
	std::copy(gc.A.get(), gc.A.get() + s, A.get());
	std::copy(gc.B.get(), gc.B.get() + s, B.get());
	std::copy(gc.C.get(), gc.C.get() + s, C.get());
	std::copy(gc.D.get(), gc.D.get() + s, D.get());
	std::copy(gc.alpha.get(), gc.alpha.get() + s, alpha.get());

#ifndef RANDOM_WB
	w = std::make_unique<float[]>(s);
	std::copy(gc.w.get(), gc.w.get() + s, w.get());
#endif

#ifdef OJA
	delta = std::make_unique<float[]>(s);
	std::copy(gc.delta.get(), gc.delta.get() + s, delta.get());
#endif


#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	std::copy(gc.gamma.get(), gc.gamma.get() + s, gamma.get());
#endif

	s = nLines;

#ifndef RANDOM_WB
	biases = std::make_unique<float[]>(s);
	std::copy(gc.biases.get(), gc.biases.get() + s, biases.get());
#endif

#ifdef STDP
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	std::copy(gc.STDP_mu.get(), gc.STDP_mu.get() + s, STDP_mu.get());
	std::copy(gc.STDP_lambda.get(), gc.STDP_lambda.get() + s, STDP_lambda.get());
#endif
}

InternalConnexion_G InternalConnexion_G::operator=(const InternalConnexion_G& gc) {

	nLines = gc.nLines;
	nColumns = gc.nColumns;

	int s = nLines * nColumns;

	eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);
	D = std::make_unique<float[]>(s);
	alpha = std::make_unique<float[]>(s);

	std::copy(gc.eta.get(), gc.eta.get() + s, eta.get());
	std::copy(gc.A.get(), gc.A.get() + s, A.get());
	std::copy(gc.B.get(), gc.B.get() + s, B.get());
	std::copy(gc.C.get(), gc.C.get() + s, C.get());
	std::copy(gc.D.get(), gc.D.get() + s, D.get());
	std::copy(gc.alpha.get(), gc.alpha.get() + s, alpha.get());

#ifndef RANDOM_WB
	w = std::make_unique<float[]>(s);
	std::copy(gc.w.get(), gc.w.get() + s, w.get());
#endif

#ifdef OJA
	delta = std::make_unique<float[]>(s);
	std::copy(gc.delta.get(), gc.delta.get() + s, delta.get());
#endif

#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	std::copy(gc.gamma.get(), gc.gamma.get() + s, gamma.get());
#endif

	s = nLines;

#ifndef RANDOM_WB
	biases = std::make_unique<float[]>(s);
	std::copy(gc.biases.get(), gc.biases.get() + s, biases.get());
#endif

#ifdef STDP
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	std::copy(gc.STDP_mu.get(), gc.STDP_mu.get() + s, STDP_mu.get());
	std::copy(gc.STDP_lambda.get(), gc.STDP_lambda.get() + s, STDP_lambda.get());
#endif

	return *this;
}


InternalConnexion_G::InternalConnexion_G(std::ifstream& is)
{
	READ_4B(nLines, is);
	READ_4B(nColumns, is);

	int s = nLines * nColumns;

	eta = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(eta.get()), s * sizeof(float));
	A = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(A.get()), s * sizeof(float));
	B = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(B.get()), s * sizeof(float));
	C = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(C.get()), s * sizeof(float));
	D = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(D.get()), s * sizeof(float));
	alpha = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(alpha.get()), s * sizeof(float));

#ifndef RANDOM_WB
	w = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(w.get()), s * sizeof(float));
#endif

#ifdef OJA
	delta = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(delta.get()), s * sizeof(float));
#endif


#ifdef CONTINUOUS_LEARNING
	gamma = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(gamma.get()), s * sizeof(float));
#endif


	s = nLines;
	
#ifndef RANDOM_WB
	biases = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(biases.get()), s * sizeof(float));
#endif


#ifdef STDP
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(STDP_mu.get()), s * sizeof(float));
	is.read(reinterpret_cast<char*>(STDP_lambda.get()), s * sizeof(float));
#endif
}

void InternalConnexion_G::save(std::ofstream& os) 
{
	WRITE_4B(nLines, os);
	WRITE_4B(nColumns, os);

	int s = nLines * nColumns;

	os.write(reinterpret_cast<const char*>(eta.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(A.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(B.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(C.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(D.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(alpha.get()), s * sizeof(float));
	
#ifndef RANDOM_WB
	os.write(reinterpret_cast<char*>(w.get()), s * sizeof(float));
#endif

#ifdef OJA
	os.write(reinterpret_cast<char*>(delta.get()), s * sizeof(float));
#endif

#ifdef CONTINUOUS_LEARNING
	os.write(reinterpret_cast<const char*>(gamma.get()), s * sizeof(float));
#endif

	s = nLines;

#ifndef RANDOM_WB
	os.write(reinterpret_cast<const char*>(biases.get()), s * sizeof(float));
#endif

#ifdef STDP
	os.write(reinterpret_cast<const char*>(STDP_mu.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(STDP_lambda.get()), s * sizeof(float));
#endif
}


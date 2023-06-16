#pragma once

#include "Trial.h"


struct GeneratorParameters {

	//defaults:
	GeneratorParameters()
	{

	}
};



struct Generator 
{
	Generator();
	~Generator();

	void step();

	void save();
	void save1Net();

	void setParameters(GeneratorParameters& params) {};
};

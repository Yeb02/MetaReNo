#pragma once


////////////////////////////////////
///// USER COMPILATION CHOICES /////
////////////////////////////////////


// Comment/uncomment or change the value of various preprocessor directives
// to compile different versions of the code. Or use the -D flag.




// When defined, wLifetime updates take place during the trial and not at the end of it. The purpose is to
// allow for a very long term memory, in parallel with E and H but much slower.
// Should be defined if there is just 1 trial, or equivalently no trials at all. Could be on even if there 
// are multiple trials. Adds the gamma array.
#define CONTINUOUS_LEARNING

// Defined if and only if there should not be any lifelong (i.e. inter trial) learning, when CONTINUOUS_LEARNING is used.
#ifdef CONTINUOUS_LEARNING
#define ZERO_WL_BEFORE_TRIAL  
#endif

// When defined, for each network, float sp = sum of a function F of each activation of the network, at each step.
// F is of the kind pow(activation, 2*k), so that its symmetric around 0 and decreasing in [-1,0]. (=> increasing in [0, -1])
// At the end of the lifetime, sp is divided by the numer of steps and the number of activations, as both may differ from
// one specimen to another. The vector of [sp/(nA*nS) for each specimen] is normalized (mean 0 var 1), and for each specimen
// the corresponding value in the vector is substracted to the score in parallel of size and amplitude regularization terms
// when computing fitness. The lower sum(F), the fitter.
#define SATURATION_PENALIZING

// At each inference, a small random proportion of the lifetime quantities is reset to either 0 or a random value. These are,
// when available; wL, H, E, w.
#define DROPOUT

#define MODULATION_VECTOR_SIZE 2     // DO NOT CHANGE

	
// When defined, presynaptic activities of complexNodes (topNode excepted) are an exponential moving average. Each node 
// be it Modulation, complex, memory or output has an evolved parameter (STDP_decay) that parametrizes the average.
// WARNING only compatible with N_ACTIVATIONS = 1, I havent implemented all the derivatives in complexNode_P::forward yet
//#define STDP

// The fixed weights w and biases b are not computed by the meta-nets, but set randomly  (w -> uniform(-.1,.1), b->NORMAL).
// This reset happens either at the beginning of each trial, or lifetime. 
#define RANDOM_WB

// Adds Oja's rule to the ABCD rule. This requires the addition of the matrices delta and storage_delta to InternalConnexion_G, 
// deltas being in the [0, 1] range. The update of E is now :  E = (1-eta)E + eta(ABCD... - delta*yj*yj*w_eff),  where w_eff is the 
// effective weight, something like w_eff = w + alpha * H + wL. 
//#define OJA



/////////////////////////// CALCULATIONS, IGNORE EVERYTHING FROM HERE ONWARDS ////////////////////////////////////////////////////

#ifdef CONTINUOUS_LEARNING
#define CL_MAT 1 // gamma
#else
#define CL_MAT 0
#endif 

#ifdef STDP
#define STDP_ARR 2 // lambda, mu
#else
#define STDP_ARR 0
#endif 

#ifdef RANDOM_WB
#define RWB_MAT 0 
#define RWB_ARR 0 
#else
#define RWB_MAT 1 // w
#define RWB_ARR 1 // b
#endif 

#ifdef OJA
#define OJA_MAT 1 // delta
#else
#define OJA_MAT 0
#endif 


// How many matrices there are per complex node.  (each of size nLines*nColumns)
// The 6 accounts for A,B,C,D,eta,alpha
#define N_MATRICES (6 + OJA_MAT + RWB_MAT + CL_MAT)

// How many arrays there are per complex node. (each of size nLines)
#define N_ARRAYS (RWB_ARR + STDP_ARR)
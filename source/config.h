#pragma once


////////////////////////////////////
///// USER COMPILATION CHOICES /////
////////////////////////////////////


// Comment/uncomment or change the value of various preprocessor directives
// to compile different versions of the code. Or use the -D flag.




// When defined, wLifetime updates take place during the trial and not at the end of it. The purpose is to
// allow for a very long term memory, in parallel with E and H but much slower.
// Should be defined if there is just 1 trial, or equivalently no trials at all. Could be on even if there 
// are multiple trials. 
#define CONTINUOUS_LEARNING


// When defined, for each network, float sp = sum of a function F of each activation of the network, at each step.
// F is of the kind pow(activation, 2*k), so that its symmetric around 0 and decreasing in [-1,0]. (=> increasing in [0, -1])
// At the end of the lifetime, sp is divided by the numer of steps and the number of activations, as both may differ from
// one specimen to another. The vector of [sp/(nA*nS) for each specimen] is normalized (mean 0 var 1), and for each specimen
// the corresponding value in the vector is substracted to the score in parallel of size and amplitude regularization terms
// when computing fitness. The lower sum(F), the fitter.
#define SATURATION_PENALIZING


#define MODULATION_VECTOR_SIZE 2     // DO NOT CHANGE


// TODO : implement DERIVATOR
// CENTERED_TANH(x) = tanhf(x) * expf(-x*x) * 1/.375261
// I dont really know what to expect from non-monotonous functions when it comes to applying 
// hebbian updates... It does not make much sense. But I plan to add cases where activations
// do not use hebbian rules.
#define N_ACTIVATIONS  1 // only activation functions < N_ACTIVATIONS are used
const enum ACTIVATION { TANH = 0, GAUSSIAN = 1, LOG2 = 2, EXP2 = 3, RELU = 4, SINE = 5, CENTERED_TANH = 6 };

	
// When defined, presynaptic activities of complexNodes (topNode excepted) are an exponential moving average. Each node 
// be it Modulation, complex, memory or output has an evolved parameter (STDP_decay) that parametrizes the average.
// WARNING only compatible with N_ACTIVATIONS = 1, I havent implemented all the derivatives in complexNode_P::forward yet
#define STDP

// The fixed weights w are not evolved anymore, but set randomly (uniform(-.1,.1)) at the beginning of each trial. This also means
// that it is now InternalConnexion_P and not InternalConnexion_G that handles the w matrix. When used in conjunction with
// GUIDED_MUTATIONS, w from ordinary connexions are no longer used. 
#define RANDOM_W

// Adds Oja's rule to the ABCD rule. This requires the addition of the matrices delta and storage_delta to InternalConnexion_G, 
// deltas being in the [0, 1] range. The update of E is now :  E = (1-eta)E + eta(ABCD... - delta*yj*yj*w_eff),  where w_eff is the 
// effective weight, something like w_eff = w + alpha * H + wL. 
#define OJA
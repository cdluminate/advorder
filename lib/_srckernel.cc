/* Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
 * Released under the Apache-2.0 License.

 * C++ Implementation of Short-Range Ranking Correlation.
 * Significantly faster than the primitive implementation in python/numpy.
 *
 [cProfile]

 --> Fashion, K=5, Q=1000, T=100.

 // Python -> 24.578 :: 100%
	1    0.012    0.012   38.074   38.074 BlackOA.py:21(BlackAttack)
  101    0.415    0.004   35.915    0.356 reorder.py:442(SPSA)
 1897    0.175    0.000   24.578    0.013 reorder.py:41(BatchNearsightRankCorr) *
95052   13.964    0.000   24.374    0.000 reorder.py:56(NearsightRankCorr)

 // C++ -> 57% (logicall simplified version)
    1    0.014    0.014   27.078   27.078 BlackOA.py:21(BlackAttack)
  101    0.409    0.004   24.914    0.247 reorder.py:442(SPSA)
 1852    0.012    0.000   14.055    0.008 srckernel.py:25(BatchNearsightRankCorr) *
 1852   13.918    0.008   13.918    0.008 {built-in method SRC.BatchShortRangeRankingCorrelation}

 // C++ -> 56%
	1    0.014    0.014   26.692   26.692 BlackOA.py:21(BlackAttack)
  101    0.411    0.004   24.531    0.243 reorder.py:442(SPSA)
 1847    0.012    0.000   13.727    0.007 srckernel.py:25(BatchNearsightRankCorr)
 1847   13.593    0.007   13.593    0.007 {built-in method SRC.BatchShortRangeRankingCorrelation}

 // Rust -> 24%
 
	1    0.011    0.011   15.795   15.795 BlackOA.py:21(BlackAttack)
  101    0.364    0.004   13.630    0.135 reorder.py:445(SPSA)
 2142    0.102    0.000    7.172    0.003 reorder.py:956(__call__)
 1738    0.005    0.000    5.822    0.003 srckernel_rs.py:46(BatchNearsightRankCorr)
 1738    0.074    0.000    5.738    0.003 srckernel_rs.py:52(<listcomp>)
87102    1.166    0.000    5.682    0.000 srckernel_rs.py:39(NearsightRankCorr)

 --> Fashion, K=25, Q=1000, T=100

 // Python -> 359.729 :: 100%
        1    0.015    0.015  375.515  375.515 BlackOA.py:21(BlackAttack)
      101    0.517    0.005  373.261    3.696 reorder.py:445(SPSA)
   101202  270.129    0.003  360.211    0.004 reorder.py:56(NearsightRankCorr)
     2020    0.201    0.000  359.729    0.178 reorder.py:41(BatchNearsightRankCorr)

 // C++ -> 36%

        1    0.014    0.014  145.958  145.958 BlackOA.py:21(BlackAttack)
      101    0.502    0.005  143.762    1.423 reorder.py:445(SPSA)
     2020    0.018    0.000  130.995    0.065 srckernel.py:25(BatchNearsightRankCorr)
     2020  130.844    0.065  130.844    0.065 {built-in method SRC.BatchShortRangeRankingCorrelation}

 // Rust -> 2.3%
 
	1    0.011    0.011   20.336   20.336 BlackOA.py:21(BlackAttack)
  101    0.435    0.004   18.047    0.179 reorder.py:445(SPSA)
 2020    0.006    0.000    8.602    0.004 srckernel_rs.py:46(BatchNearsightRankCorr)
 2424    0.122    0.000    8.500    0.004 reorder.py:956(__call__)
 2020    0.094    0.000    8.498    0.004 srckernel_rs.py:52(<listcomp>)
101202    3.320    0.000    8.425    0.000 srckernel_rs.py:39(NearsightRankCorr)
 
 * reference material: https://www.cnblogs.com/yanghailin/p/12901586.html
 */
#include <iostream>
#include <math.h>
#include <map>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/torch.h>
using namespace std;

#define REWARD_CONCORDANT +1.0
#define REWARD_DISCORDANT -1.0
#define REWARD_OUTOFRANGE -1.0

float ShortRangeRankingCorrelation(
    torch::Tensor argsort,
	torch::Tensor otopk,
	torch::Tensor rperm)
{
	// dimension check
	if (argsort.dim() != 1 || otopk.dim() != 1 || rperm.dim() != 1) {
		return FP_NAN;
	}
	// shape check
	if (argsort.size(0) < rperm.size(0)) {
		return FP_NAN;
	}
	// helper constants
	auto K = rperm.size(0);
	// permute the topk results
	auto rtopk = otopk.index_select(0, rperm);
	//!cout << rtopk << endl;
	// initialize maps for quick access
	//!map<int,int> ov2ij;
	//!for (int i = 0; i < K; i++)
	//!	ov2ij.insert(pair<int,int>(otopk[i].item().toInt(), i));
	//!cout << ov2ij << endl;
	map <int,int> av2ij;
	for (int i = 0; i < argsort.size(0); i++)
		av2ij.insert(pair<int,int>(argsort[i].item().toInt(), i));
	//!cout << av2ij << endl;
	map <int,int> rv2ij;
	for (int i = 0; i < K; i++)
		rv2ij.insert(pair<int,int>(rtopk[i].item().toInt(), i));
	// initialize zero matrix
	auto scores = at::zeros({ K, K }, at::kInt);
	//!cout << scores << endl;
	// calculating tau_s score matrix : row
//#pragma omp parallel for num_threads(2) // not quite useful
	for (int i = 0; i < K; i++) {
		auto io = otopk[i].item().toInt();
		// does otopk[i] exist in the current ranking?
		auto isearch = av2ij.find(io);
		if (isearch == av2ij.end()) {
			for (int j = 0; j < i; j++)
				scores[i][j] = REWARD_OUTOFRANGE;
			continue;
		}
		// calculating tau_s score matrix : column
		for (int j = 0; j < i; j++) {
			auto jo = otopk[j].item().toInt();
			// does otopk[j] exist in the current ranking?
			auto jsearch = av2ij.find(jo);
			if (jsearch == av2ij.end()) {
			   	scores[i][j] = REWARD_OUTOFRANGE;
				continue;
			}
			// get i,j ranks in two lists
			auto cranki = av2ij[io];
			auto crankj = av2ij[jo];
			auto xranki = rv2ij[io];
			auto xrankj = rv2ij[jo];
			// case 1: concordant
			if ( ((cranki > crankj) && (xranki > xrankj)) ||
					((cranki < crankj) && (xranki < xrankj)) )
				scores[i][j] = REWARD_CONCORDANT;
			// case 2: discordant
			else if ( ((cranki > crankj) && (xranki < xrankj)) ||
					((cranki < crankj) && (xranki > xrankj)) )
				scores[i][j] = REWARD_DISCORDANT;
			/*
			// [logically simplified version]
			bool cilj = (av2ij[io] < av2ij[jo]);
			bool xilj = (rv2ij[io] < rv2ij[jo]);
			if (cilj ^ xilj) {
				scores[i][j] = REWARD_DISCORDANT;
			} else {
				scores[i][j] = REWARD_CONCORDANT;
			}
			*/
		}
	}
	//!cout << scores << endl;
	// get tau_s
	float tau_s = scores.sum().item().toFloat();
	tau_s /= ( K * (K - 1) ) / 2.0;
	return tau_s;
}

torch::Tensor BatchShortRangeRankingCorrelation(
		torch::Tensor argsort,
		torch::Tensor otopk,
		torch::Tensor rperm)
{
	auto taus = at::ones(argsort.size(0), at::kFloat);
//#pragma omp parallel for num_threads(2)
//	for (int i = 0; i < argsort.size(0); i++) {
//		taus[i] = ShortRangeRankingCorrelation(argsort[i], otopk, rperm);
//	}
// https://discuss.pytorch.org/t/using-at-parallel-for-in-a-custom-operator/82747
	at::parallel_for(0, argsort.size(0), 0, [&](int64_t begin, int64_t end){
		for (int64_t i = begin; i < end; i++) {
			taus[i] = ShortRangeRankingCorrelation(argsort[i], otopk, rperm);
		}
	});
	return taus;
}

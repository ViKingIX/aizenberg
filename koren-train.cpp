/*
 * koren-train.cpp
 * Copyright (C) 2014 ViKing <ViKingIX@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <csignal>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/timer/timer.hpp>
#include <omp.h>
#include "aizenberg.h"

using namespace std;
namespace ublas = boost::numeric::ublas;
typedef ublas::vector<double> uvec;
typedef ublas::zero_vector<double> zvec;
namespace po = boost::program_options;

unsigned niter;		//number of iterations
unsigned ncv;		//number of foldings
unsigned l;		//number of dimensions
unsigned Nslot;		//number of slots per day
unsigned w;		//size of time window (in seconds)
double lambda;		//lambda for regularization
string mode;		//brute or imp
double fraction;	//fration of training data
unsigned Jmaxlen = 1000;
unsigned nproc;
double vmax = 1;
double vmin = -1;
unsigned nfields = 6;	//number of fields to be recognized as data
const unsigned SEC_PER_DAY = 24 * 60 * 60, SEC_PER_HOUR = 60 * 60;
const string prog_name = "koren-train";
vector<unsigned> stations;
bool breakflag = false;

void print_usage(const po::options_description &desc, const po::positional_options_description &pd)/*{{{*/
{
	cout << "Usage: " << prog_name << " [options] logfile [modelfile]" << endl << endl;
	cout << desc << endl;
	return;
}/*}}}*/

inline double clip(double x, double low = vmin, double high = vmax)/*{{{*/
{
	return min(max(x, vmin), vmax);
}/*}}}*/

inline uvec vclip(uvec v, double low = vmin, double high = vmax)/*{{{*/
{
	for (int i = 0;i < v.size();i ++)
		v[i] = min(max(v[i], vmin), vmax);
	return v;
}/*}}}*/

void sgd(const map<unsigned, vector<example> > &D, const map<unsigned, unsigned> &a, const vector<unsigned> &S, const map<unsigned, unsigned> &Scount, const vector<unsigned> &tras, double eta, string mode, vector<unsigned> &J, map<unsigned, unsigned> &Jcount, Theta &theta)/*{{{*/
{
	static vector<unsigned> random_s;
	map<unsigned, double> &Ca = theta.Ca, &C = theta.C;
	map<unsigned, uvec> &Pa = theta.Pa, &P = theta.P, &V = theta.V;
	map<unsigned, vector<uvec> > &Vt = theta.Vt;
	ublas::zero_vector<double> z(theta.l);
	
	if (!random_s.size())
	{
		random_s.resize(D.size());
		for (int i = 0;i < D.size();i ++)
			random_s[i] = i;
	}

	random_shuffle(random_s.begin(), random_s.end());
	for (const int &s : random_s)
	{
		const vector<example> &Ps = D.at(s);
		vector<example> Pstw;

		for (const auto &exi : Ps)
		{
			boost::timer::auto_cpu_timer ct("example takes %ws\n");
			map<unsigned, double> dCa, dC;
			map<unsigned, uvec> dPa, dP;
			uvec dV, dVt;

			uvec qsum = z;
			//remove j not in (t - w, t] and calculate qsum
			for (auto it = Pstw.begin();it != Pstw.end();)
				if (it->t < exi.t && exi.t - it->t >= w)
					it = Pstw.erase(it);
				else
				{
					unsigned aj = it->art, j = it->tra;
					if (!dPa.count(aj))
						dPa[aj] = z;
					if (!dP.count(j))
						dP[j] = z;
					qsum += Pa[aj] + P[j];
					it ++;
				}

			double coeff = 0;
			if (Pstw.size() > 0)
				coeff = 1. / sqrt(Pstw.size());
			unsigned ai = exi.art, i = exi.tra;
			double bi = Ca[ai] + C[i];
			uvec qi = Pa[ai] + P[i];
			unsigned slot = exi.t % SEC_PER_DAY / SEC_PER_HOUR / Nslot;
			uvec vterm = V[s] + Vt[s][slot] + coeff * qsum;
#ifdef DEBUG
			cout << "bi: " << bi << endl;
			cout << "qi: " << qi << endl;
			cout << "vs: " << V[s] << endl;
			cout << "vs_t: " << Vt[s][slot] << endl;
			cout << "qsum: " << coeff * qsum << endl;
			cout << "vterm: " << vterm << endl;
#endif

			dCa[ai] = dC[i] = eta;
			dPa[ai] = dP[i] = eta * vterm;
			dV = dVt = eta * qi;
			for (const auto &exj : Pstw)
			{
				unsigned aj = exj.art, j = exj.tra;
				dPa[aj] += eta * coeff * qi;
				dP[j] += eta * coeff * qi;
			}

			//rsj;t part
			if (mode == "brute")/*{{{*/
			{
#ifdef DEBUG
				boost::timer::auto_cpu_timer ct("brute force costs %ws\n");
#endif
				map<int, double> expr;
				double denom = 0;
				for (unsigned j : tras)
				{
					unsigned aj = a.at(j);
					if (!dCa.count(aj))
						dCa[aj] = 0;
					if (!dC.count(j))
						dC[j] = 0;
					if (!dPa.count(aj))
						dPa[aj] = z;
					if (!dP.count(j))
						dP[j] = z;
					double bj = Ca[aj] + C[j];
					uvec qj = Pa[aj] + P[j];
					double exprj = exp(bj + inner_prod(qj, vterm));
					//test exprj/*{{{*/
					bool flag = false;
					if (std::isnan(exprj))
					{
						cerr << "NaN exprj founded!" << endl;
						flag = true;
					}
					if (std::isinf(exprj))
					{
						cerr << "inf exprj!\n";
						flag = true;
					}
					if (exprj == 0)
					{
						cerr << "0 exprj\n";
						flag = true;
					}
					if (flag)
					{
						cerr << "bj: " << bj << endl;
						cerr << "qj: " << qj << endl;
						cerr << "vterm: " << vterm << endl;
						continue;
					}/*}}}*/
					expr[j] = exprj;
					denom += exprj;
				}

				for (unsigned j : tras)
				{
					unsigned aj = a.at(j);
					uvec qj = Pa[aj] + P[j];
					double coeff2 = -eta * expr[j] / denom;
					if (std::isnan(coeff2))/*{{{*/
					{
						cerr << "NaN coeff2!\n";
						cerr << "eta: " << eta << endl;
						cerr << "exprj: " << expr[j] << endl;
						cerr << "denom: " << denom << endl;
						continue;
					}/*}}}*/
					dCa[aj] += coeff2;
					dC[j] += coeff2;
					dPa[aj] += coeff2 * vterm;
					dP[j] += coeff2 * vterm;
					dV += coeff2 * qj;
					dVt += coeff2 * qj;
					for (const auto &exk : Pstw)
					{
						unsigned k = exk.tra, ak = exk.art;
						dPa[ak] += coeff2 * coeff * qj;
						dP[k] += coeff2 * coeff * qj;
					}
				}
			}/*}}}*/
			else if (mode == "imp")/*{{{*/
			{
				boost::timer::auto_cpu_timer ct("importance sampling costs %ws\n");
				double Jsum = 0, denom = 0;
				map<unsigned, double> expr;
				{
					boost::timer::auto_cpu_timer ct("compute Jsum takes %ws\n");
//#pragma omp parallel for reduction(+:denom) reduction(+:Jsum)
				for (int x = 0;x < J.size();x ++)
				{
					unsigned j = J.at(x), aj = a.at(j);
					double bj = Ca.at(aj) + C.at(j);
					uvec qj = Pa.at(aj) + P.at(j);
//#pragma omp critical
					{
						if (!expr.count(j))
							expr[j] = exp(bj + inner_prod(qj, vterm));
					}
					denom += expr.at(j) * S.size() / Scount.at(j);
					Jsum += expr.at(j);
				}
				}
				double expri = exp(bi + inner_prod(qi, vterm));
				{
					boost::timer::auto_cpu_timer ct("sampling takes %ws\n");
				while (Jsum <= expri && J.size() < Jmaxlen)
				{
					unsigned x = rand() % S.size();
					unsigned j = S[x], aj = a.at(j);
					double bj = Ca[aj] + C[j];
					uvec qj = Pa[aj] + P[j];
					if (!expr.count(j))
						expr[j] = exp(bj + inner_prod(qj, vterm));
					//test exprj/*{{{*/
					double exprj = expr[j];
					bool flag = false;
					if (std::isnan(exprj))
					{
						cerr << "NaN exprj founded!" << endl;
						flag = true;
					}
					if (std::isinf(exprj))
					{
						cerr << "inf exprj!\n";
						flag = true;
					}
					if (exprj == 0)
					{
						cerr << "0 exprj\n";
						flag = true;
					}
					if (flag)
					{
						cerr << "bj: " << bj << endl;
						cerr << "qj: " << qj << endl;
						cerr << "vterm: " << vterm << endl;
						continue;
					}/*}}}*/
					denom += expr[j] * S.size() / Scount.at(j);
					Jsum += expr[j];
					J.push_back(S[x]);
				}
				}
				{
					boost::timer::auto_cpu_timer ct("gradient takes %ws\n");
				for (unsigned j : J)
				{
					unsigned aj = a.at(j);
					double coeff2 = -eta * expr[j] * S.size() / Scount.at(j) / denom;
					uvec qj = P[j] + Pa[aj];
					if (!dCa.count(aj))
						dCa[aj] = 0;
					if (!dC.count(j))
						dC[j] = 0;
					if (!dPa.count(aj))
						dPa[aj] = z;
					if (!dP.count(j))
						dP[j] = z;
					dCa[aj] += coeff2;
					dC[j] += coeff2;
					dPa[aj] += coeff2 * vterm;
					dP[j] += coeff2 * vterm;
					dV += coeff2 * qj;
					dVt += coeff2 * qj;
					for (const auto &exk : Pstw)
					{
						unsigned ak = exk.art, k = exk.tra;
						dPa[ak] += coeff2 * coeff * qj;
						dP[k] += coeff2 * coeff * qj;
					}
				}
				}
			}/*}}}*/

			//update/*{{{*/
#pragma omp sections
			{
#pragma omp section
				{
					for (auto it = dCa.cbegin();it != dCa.cend();it ++)
					{
#ifdef DEBUG
						cout << "dCa[" << it->first << "] = " << it->second << endl;
#endif
						Ca[it->first] = clip(Ca[it->first] + it->second - eta * 2 * lambda * Ca[it->first], -1, 1);
					}
				}
#pragma omp section
				{
					for (auto it = dC.cbegin();it != dC.cend();it ++)
					{
#ifdef DEBUG
						cout << "dC[" << it->first << "] = " << it->second << endl;
#endif
						C[it->first] = clip(C[it->first] + it->second - eta * 2 * lambda * C[it->first], -1, 1);
					}
				}
#pragma omp section
				{
					for (auto it = dPa.cbegin();it != dPa.cend();it ++)
					{
#ifdef DEBUG
						cout << "dPa[" << it->first << "] = " << it->second << endl;
#endif
						Pa[it->first] = vclip(Pa[it->first] + it->second - eta * 2 * lambda * Pa[it->first], -1, 1);
					}
				}
#pragma omp section
				{
					for (auto it = dP.cbegin();it != dP.cend();it ++)
					{
#ifdef DEBUG
						cout << "dP[" << it->first << "] = " << it->second << endl;
#endif
						P[it->first] = vclip(P[it->first] + it->second - eta * 2 * lambda * P[it->first], -1, 1);
					}
				}
			}
#ifdef DEBUG
			cout << "dV[" << s << "] = " << dV << endl;
#endif
			V[s] = vclip(V[s] + dV - eta * 2 * lambda * V[s], -1, 1);
#ifdef DEBUG
			cout << "dVt[" << s << "][" << slot << "] = " << dVt << endl;
			cout << endl;
#endif
			Vt[s][slot] = vclip(Vt[s][slot] + dVt - eta * 2 * lambda * Vt[s][slot], -1, 1);
			/*}}}*/
			
			Pstw.push_back(exi);
		}
	}
	return;
}/*}}}*/

struct delta
{
	map<unsigned, double> dCa, dC;
	map<unsigned, uvec> dPa, dP, dV;
	map<unsigned, vector<uvec> > dVt;
};

delta gd_body(unsigned tid, unsigned s, const vector<example> &Ps, const Theta &theta)
{
	delta d;
	return d;
}

void gd(const map<unsigned, vector<example> > &D, const map<unsigned, unsigned> &a, const vector<unsigned> &S, const vector<unsigned> &Scount, double eta, string mode, map<unsigned, vector<example> > &J, Theta &theta)/*{{{*/
{
	vector<delta> Delta(nproc);
	map<unsigned, double> &Ca = theta.Ca, C = theta.C;
	map<unsigned, uvec> &Pa = theta.Pa, P = theta.P, V = theta.V;
	map<unsigned, vector<uvec> > &Vt = theta.Vt;
	ublas::zero_vector<double> z(theta.l);
#pragma omp parallel for num_threads(nproc)
	for (int x = 0;x < stations.size();x ++)
	{
		unsigned tid = omp_get_thread_num();
		unsigned s = stations[x];
		const vector<example> &Ps = D.at(s);
		vector<example> Pstw;
		map<unsigned, double> &dCa = Delta[tid].dCa, &dC = Delta[tid].dC;
		map<unsigned, uvec> &dPa = Delta[tid].dPa, &dP = Delta[tid].dP, &dV = Delta[tid].dV;
		map<unsigned, vector<uvec> > &dVt = Delta[tid].dVt;
		for (const auto &exi : Ps)
		{
			uvec qsum = z;
			for (auto it = Pstw.begin();it != Pstw.end();)
			{
				const example &exj = *it;
				unsigned aj = it->art, j = it->tra;
				if (exj.t < exi.t && exi.t - exj.t < w)
				{
					if (!dPa.count(aj))
						dPa[aj] = z;
					if (!dP.count(j))
						dP[j] = z;
					qsum += P.at(j) + Pa.at(aj);
					it++;
				}
				else
					it = Pstw.erase(it);
			}
			unsigned ai = exi.art, i = exi.tra;
			uvec qi = P.at(i) + Pa.at(ai);
			double coeff = 0;
			if (Pstw.size() > 0)
				coeff = 1. / Pstw.size();
			unsigned slot = exi.t % 86400 / 60 / 60 / Nslot;
			uvec vterm = V.at(s) + Vt.at(s)[slot] + coeff * qsum;
			if (!dCa.count(ai))
				dCa[ai] = 0;
			if (!dC.count(i))
				dC[i] = 0;
			if (!dPa.count(ai))
				dPa[ai] = z;
			if (!dP.count(i))
				dP[i] = z;
			if (!dV.count(s))
				dV[s] = z;
			if (!dVt.count(s))
				dVt[s] = vector<uvec>(Nslot, z);
			dCa[ai] += 1;
			dC[i] += 1;
			dPa[ai] += vterm;
			dP[i] += vterm;
			dV[s] += qi;
			dVt[s][slot] += qi;
			for (const auto &exj : Pstw)
			{
				unsigned aj = exj.art, j = exj.tra;
				if (!dPa.count(aj))
					dPa[aj] = z;
				if (!dP.count(j))
					dP[j] = z;
				dPa[aj] += coeff * qi;
				dP[j] += coeff * qi;
			}

			map<unsigned, double> expr;
			double denom = 0, Jsum = 0;
			vector<unsigned> J;
			for (int x = 0;x < 128;x ++)
				J.push_back(S.at(rand() % S.size()));
			for (unsigned j : J)
			{
				unsigned aj = a.at(j);
				double bj = C[j] + Ca[aj];
				uvec qj = P[j] + Pa[aj];
				if (!expr.count(j))
					expr[j] = exp(bj + inner_prod(qj, vterm));
				denom += expr[j] * S.size() / Scount.at(j);
			}
			for (unsigned j : J)
			{
				unsigned aj = a.at(j);
				double coeff2 = -expr[j] * S.size() / Scount.at(j) / denom;
				uvec qj = P[j] + Pa[j];
				if (!dCa.count(aj))
					dCa[aj] = 0;
				if (!dC.count(j))
					dC[j] = 0;
				if (!dPa.count(aj))
					dPa[aj] = z;
				if (!dP.count(j))
					dP[j] = z;
				dCa[aj] += coeff2;
				dC[j] += coeff2;
				dPa[aj] += coeff2 * vterm;
				dP[j] += coeff2 * vterm;
				for (const auto &exk : Pstw)
				{
					unsigned ak = exk.art, k = exk.tra;
					dPa[ak] += coeff2 * qj;
					dP[k] += coeff2 * qj;
				}
			}
			
			Pstw.push_back(exi);
		}
	}
	for (const auto &d : Delta)
	return;
}/*}}}*/

double cal_L(const map<unsigned, vector<example> > &D, const map<unsigned, unsigned> &a, const vector<unsigned> &tras, const Theta &theta)/*{{{*/
{
	boost::timer::auto_cpu_timer ct("compute likelihood takes %ws\n");
	long double L = 0;
#pragma omp parallel for reduction(+:L)
	for (int x = 0;x < stations.size();x ++)
	{
		unsigned s = stations[x];
		const vector<example> &Ps = D.at(s);
		vector<example> Pstw;
		
		for (const example &exi : Ps)
		{
			unsigned ai = exi.art, i = exi.tra;
			double bi = theta.Ca.at(ai) + theta.C.at(i);
			uvec qi = theta.Pa.at(ai) + theta.P.at(i);
			uvec qsum = zvec(theta.l);

			for (auto it = Pstw.begin();it != Pstw.end();)
				if (it->t < exi.t && exi.t - it->t >= w)
					it = Pstw.erase(it);
				else
				{
					qsum += theta.Pa.at(it->art) + theta.P.at(it->tra);
					it ++;
				}
			double coeff = 0;
			if (Pstw.size() > 0)
				coeff = 1. / sqrt(Pstw.size());
			unsigned slot = exi.t % 86400 / 60 / 60 / theta.Nslot;
			uvec vterm = theta.V.at(s) + theta.Vt.at(s)[slot] + coeff * qsum;

			double denom = 0;
			double expri = exp(bi + inner_prod(qi, vterm));
			for (unsigned j : tras)
			{
				unsigned aj = a.at(j);
				double bj = theta.Ca.at(aj) + theta.C.at(j);
				uvec qj = theta.Pa.at(aj) + theta.P.at(j);
				denom += exp(bj + inner_prod(qj, vterm));
			}
#if 0
			cout << "bi: " << bi << endl;
			cout << "qi: " << qi << endl;
			cout << "vterm: " << vterm << endl;
			cout << "v: " << theta.V.at(s) << endl;
			cout << "vt: " << theta.Vt.at(s).at(slot) << endl;
			cout << "qsum: " << coeff * qsum << endl;
			cout << "expri: " << expri << ", denom: " << denom << endl << endl;
#endif
			L += log(expri / denom);

			Pstw.push_back(exi);
		}
	}
	return L;
}/*}}}*/

void INThandler(int signum)/*{{{*/
{
	breakflag = true;
	return;
}/*}}}*/

int main(int argc, const char *argv[])
{
	//Parsing arguments/*{{{*/
	po::options_description desc("Available options");
	desc.add_options()
		("help,h", "show this help message")
		("cv,v", po::value<unsigned>(&ncv)->default_value(0), "set the number of folding for cross validation")
		("eta", po::value<double>(), "set the learning rate eta")
		("fraction,f", po::value<double>(&fraction)->default_value(0.75), "set the fraction of training data")
		("iter,it", po::value<unsigned>(&niter)->default_value(20), "set the number of iterations")
		(",l", po::value<unsigned>(&l)->default_value(20), "set the dimension of latent space")
		("lambda", po::value<double>(&lambda)->default_value(1e-4), "set the weight decay constant")
		("mode,m", po::value<string>(&mode)->default_value("imp"), "set the mode when processing Pst, options are \"brute\", \"imp\"")
		("nproc", po::value<unsigned>(&nproc)->default_value(16), "set the number of threads")
		("nslot", po::value<unsigned>(&Nslot)->default_value(8), "set the number of slots per day")
		("oiter", po::bool_switch()->default_value(false), "set if need to output every iteration")
		(",w", po::value<unsigned>(&w)->default_value(30 * 60), "set the time window size for short term history (in seconds)")
		("logfile", po::value<string>()->required(), "path to input logfile")
		("modelfile", po::value<string>(), "path to output modelfile")
	;
	po::positional_options_description pd;
	pd.add("logfile", 1)
	  .add("modelfile", 1);

	po::variables_map vm;
	try
	{
		po::store(po::command_line_parser(argc, argv)
				.options(desc)
				.positional(pd)
				.run(), vm);
		if (vm.count("help") || !vm.count("logfile"))
		{
			print_usage(desc, pd);
			return 1;
		}
		po::notify(vm);
	}
	catch (exception &e)
	{
		print_usage(desc, pd);
		return 1;
	}
/*}}}*/

	signal(SIGQUIT, INThandler);

	//Load input
	map<string, unsigned> uids, artids, traids;
	map<unsigned, unsigned> a, Scount;
	vector<unsigned> S, tras;
	map<unsigned, vector<example> > D;
	unsigned Ns, Na, Nt;
	load_tsv(vm["logfile"].as<string>().c_str(), uids, artids, traids, a, D);
	for (const auto &kv : traids)
	{
		tras.push_back(kv.second);
		Scount[kv.second] = 0;
	}
	Ns = uids.size();
	Na = artids.size();
	Nt = traids.size();
	cout << "Ns = " << Ns << ", Na = " << Na << ", Nt = " << Nt << endl;
	map<unsigned, vector<example> > Dtr;
	for (auto const &it : D)
	{
		unsigned s = it.first;
		const vector<example> &Ps = it.second;
		size_t len = std::ceil(Ps.size() * fraction);
		Dtr[s] = vector<example>(Ps.begin(), Ps.begin() + len);
		for (const auto &exj : Dtr[s])
		{
			S.push_back(exj.tra);
			Scount[exj.tra]++;
		}
		stations.push_back(s);
	}

	//Initialize parameters
	Theta theta(uids, artids, traids, Nslot, l);
	theta.init();
	vector<unsigned> J;
	map<unsigned, unsigned> Jcount;

	//Training iterations
	unsigned k = 0;
	double eps = 1e-5;
	while (1)
	{
		boost::timer::auto_cpu_timer ct("iteration takes %ws\n");
		double eta = 5e-3 / (k + 1);
		if (vm.count("eta"))
			eta = vm["eta"].as<double>();

		sgd(Dtr, a, S, Scount, tras, eta, mode, J, Jcount, theta);
		if (vm["oiter"].as<bool>() || vm.count("modelfile"))
		{
			ostringstream oss;
			oss << vm["modelfile"].as<string>() << ".iter." << k + 1;
			cout << "save to " << oss.str() << endl;
			theta.saves(oss.str());
		}
		k ++;
		if (breakflag || (niter > 0 && k >= niter))
			break;
	}

	if (vm.count("modelfile"))
		theta.saves(vm["modelfile"].as<string>().c_str());
	return 0;
}

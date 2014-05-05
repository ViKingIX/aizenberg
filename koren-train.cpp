/*
 * koren-train.cpp
 * Copyright (C) 2014 ViKing <ViKingIX@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/timer/timer.hpp>
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
double fraction;	//
double vmax = 1;
double vmin = -1;
unsigned nfields = 6;	//number of fields to be recognized as data
const unsigned SEC_PER_DAY = 24 * 60 * 60, SEC_PER_HOUR = 60 * 60;
const string prog_name = "koren-train";

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

inline uvec vclip(uvec v, double low = -1, double high = 1)/*{{{*/
{
	for (int i = 0;i < v.size();i ++)
		v[i] = min(max(v[i], vmin), vmax);
	return v;
}/*}}}*/

bool sgd(const map<unsigned, vector<example> > &D, const map<unsigned, unsigned> &a, const vector<unsigned> &S, uvec &Ca, uvec &C, vector<uvec> &Pa, vector<uvec> &P, vector<uvec> &V, vector<vector<uvec> > &Vt, vector<vector<unsigned> > &J, double eta, string mode = "brute")/*{{{*/
{
	bool converge = false;
	static vector<unsigned> random_s;
	
	if (!random_s.size())
	{
		random_s.resize(D.size());
		for (int i = 0;i < D.size();i ++)
			random_s[i] = i;
	}
	random_shuffle(random_s.begin(), random_s.end());
	for (int x = 0;x < random_s.size();x ++)
	{
		unsigned s = random_s[x];
		vector<example> Ps = D.find(s)->second, Pst, Pstw;

		for (auto iti = Ps.cbegin();iti != Ps.cend();iti ++)
		{
			const example &exi = *iti;
			uvec qsum = zvec(l);
			map<unsigned, double> dCa, dC;
			map<unsigned, uvec> dPa, dP;
			uvec dV, dVt;

			//remove j not in (t - w, t] and calculate qsum
			for (auto it = Pstw.begin();it != Pstw.end();)
				if (it->t < exi.t && exi.t - it->t >= w)
					it = Pstw.erase(it);
				else
				{
					if (!dCa.count(it->art))
							dCa[it->art] = 0;
					if (!dC.count(it->tra))
							dC[it->tra] = 0;
					if (!dPa.count(it->art))
						dPa[it->art] = zvec(l);
					if (!dP.count(it->tra))
						dP[it->tra] = zvec(l);
					qsum += Pa[exi.art] + P[exi.tra];
					it ++;
				}

			double coeff = 0;
			if (Pstw.size())
				coeff = 1 / sqrt(Pstw.size());
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
			for (auto it = Pstw.cbegin();it != Pstw.cend();it ++)
			{
				unsigned aj = it->art, j = it->tra;
				if (j == i)
				{
					dPa[aj] += eta * coeff * qi;
					dP[j] += eta * coeff * qi;
				}
			}

			//rsj;t part
			if (mode == "brute")/*{{{*/
			{
				//boost::timer::auto_cpu_timer ct("brute force costs %ws\n");
				vector<example> Pst;
				for (auto it = Ps.cbegin();it != Ps.cend();it ++)
				{
					const example &exj = *it;
					if (exj.t % 86400 / 60 == exi.t % 86400 / 60)
					{
						if (!dCa.count(exj.art))
							dCa[exj.art] = 0;
						if (!dC.count(exj.tra))
							dC[exj.tra] = 0;
						if (!dPa.count(exj.art))
							dPa[exj.art] = zvec(l);
						if (!dP.count(exj.tra))
							dP[exj.tra] = zvec(l);
						Pst.push_back(exj);
					}
				}

				map<int, double> expr;
				double denom = 0;
				for (auto itj = Pst.cbegin();itj != Pst.cend();itj ++)
				{
					unsigned aj = itj->art, j = itj->tra;
					double bj = Ca[aj] + C[j];
					uvec qj = Pa[aj] + P[j];
					if (expr.count(j))
					{
						denom += expr[j];
						continue;
					}
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
				for (auto itj = Pst.cbegin();itj != Pst.cend();itj ++)
				{
					unsigned aj = itj->art, j = itj->tra;
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
					if (j == i)
					{
						dCa[aj] += coeff2;
						dC[j] += coeff2;
						dPa[aj] += coeff2 * vterm;
						dP[j] += coeff2 * vterm;
					}
					dV += coeff2 * qj;
					dVt += coeff2 * qj;
					for (auto itk = Pstw.begin();itk != Pstw.end();itk ++)
					{
						unsigned k = itk->tra, ak = itk->art;
						if (k == i)
						{
							dPa[ak] += coeff2 * coeff * qj;
							dP[k] += coeff2 * coeff * qj;
						}
					}
				}
			}/*}}}*/
			else if (mode == "imp")/*{{{*/
			{
				boost::timer::auto_cpu_timer ct("importance sampling costs %ws\n");
#if 1
				const size_t Jmaxlen = 100;
				vector<unsigned> &Js = J[s];
				double Jsum = 0, denom = 0;
				map<unsigned, double> expr;
#if 0
				vector<double> denomv(Js.size()), exprv(Js.size());
#pragma omp parallel for
				for (int x = 0;x < Js.size();x ++)
				{
					unsigned j = Js[x], aj = a.find(x)->second;
					double bj = Ca[aj] + C[j];
					uvec qj = Pa[aj] + P[j];
					double exprj = bj + inner_prod(qj, vterm);
					denomv[x] = exprj * S.size() / count(S.begin(), S.end(), j);
					expr[j] = exprj;
					Jsum[x] = exprj;
				}

#else
				for (auto it = Js.begin();it != Js.end();it ++)
				{
					unsigned aj = a.find(*it)->second, j = *it;
					double bj = Ca[aj] + C[j];
					uvec qj = Pa[aj] + P[j];
					if (!expr.count(j))
						expr[j] = bj + inner_prod(qj, vterm);
					denom += expr[j] * S.size() / count(S.begin(), S.end(), j);
					Jsum += expr[j];
				}
#endif
				double expri = exp(bi + inner_prod(qi, vterm));
				while (Jsum <= expri && Js.size() < Jmaxlen)
				{
					unsigned x = rand() % S.size();
					Js.push_back(S[x]);
					unsigned j = S[x], aj = a.find(j)->second;
					double bj = Ca[aj] + C[j];
					uvec qj = Pa[aj] + P[j];
					if (!expr.count(j))
						expr[j] = bj + inner_prod(qj, vterm);
					denom += expr[j] * S.size() / count(S.begin(), S.end(), j);
					Jsum += expr[j];
				}
				for (auto itj = Js.cbegin();itj != Js.cend();itj ++)
				{
					unsigned aj = a.find(*itj)->second, j = *itj;
					uvec qj = Pa[aj] + P[j];
					double coeff2 = -eta * expr[j] * S.size() / count(S.cbegin(), S.cend(), j) / denom;
					if (std::isnan(coeff2))/*{{{*/
					{
						cerr << "NaN coeff2!\n";
						cerr << "eta: " << eta << endl;
						cerr << "exprj: " << expr[j] << endl;
						cerr << "denom: " << denom << endl;
						continue;
					}/*}}}*/
					if (j == i)
					{
						dCa[aj] += coeff2;
						dC[j] += coeff2;
						dPa[aj] += coeff2 * vterm;
						dP[j] += coeff2 * vterm;
					}
					dV += coeff2 * qj;
					dVt += coeff2 * qj;
					for (auto itk = Pstw.begin();itk != Pstw.end();itk ++)
					{
						unsigned k = itk->tra, ak = itk->art;
						if (k == i)
						{
							dPa[ak] += coeff2 * coeff * qj;
							dP[k] += coeff2 * coeff * qj;
						}
					}
				}
#endif
			}/*}}}*/

			//update/*{{{*/
			for (auto it = dCa.cbegin();it != dCa.cend();it ++)
			{
#ifdef DEBUG
				cout << "dCa[" << it->first << "] = " << it->second << endl;
#endif
				Ca[it->first] = clip(Ca[it->first] + it->second - eta * 2 * lambda * Ca[it->first], -1, 1);
			}
			for (auto it = dC.cbegin();it != dC.cend();it ++)
			{
#ifdef DEBUG
				cout << "dC[" << it->first << "] = " << it->second << endl;
#endif
				C[it->first] = clip(C[it->first] + it->second - eta * 2 * lambda * C[it->first], -1, 1);
			}
			for (auto it = dPa.cbegin();it != dPa.cend();it ++)
			{
#ifdef DEBUG
				cout << "dPa[" << it->first << "] = " << it->second << endl;
#endif
				Pa[it->first] = vclip(Pa[it->first] + it->second - eta * 2 * lambda * Pa[it->first], -1, 1);
			}
			for (auto it = dP.cbegin();it != dP.cend();it ++)
			{
#ifdef DEBUG
				cout << "dP[" << it->first << "] = " << it->second << endl;
#endif
				P[it->first] = vclip(P[it->first] + it->second - eta * 2 * lambda * P[it->first], -1, 1);
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
	return converge;
}/*}}}*/

void load_dat(const char *logfilefn, map<unsigned, vector<example> > &D, unsigned &Ns, unsigned &Na, unsigned &Nt)/*{{{*/
{
	ifstream logfile(logfilefn, ios::in);
	if (!logfile)
		throw runtime_error("logfile not found!");
	string line;
	while (getline(logfile, line))
	{
		istringstream ss(line);
		unsigned s, i, ai;
		time_t t;
		ss >> s >> i >> ai >> t;
		if (ss.fail())
		{
			cerr << "bad format for log: " << line << endl;
			continue;
		}
		Ns = max(Ns, s + 1);
		Na = max(Na, ai + 1);
		Nt = max(Nt, i + 1);
		D[s].push_back(example(s, i, ai, t));
	}
	logfile.close();
	return;
}/*}}}*/

bool init_theta(uvec &Ca, uvec &C, vector<uvec> &Pa, vector<uvec> &P, vector<uvec> &V, vector<vector<uvec> > &Vt, int Na, int Nt, int Ns)/*{{{*/
{
	boost::timer::auto_cpu_timer ct("init_theta costs %ws\n");
#pragma omp parallel for
	for (int i = 0;i < Nt;i ++)
	{
		C[i] = 2 * (double)rand() / RAND_MAX - 1;
		for (int j = 0;j < l;j ++)
			P[i][j] = 2 * (double)rand() / RAND_MAX - 1;
	}
#pragma omp parallel for
	for (int i = 0;i < Na;i ++)
	{
		Ca[i] = 2 * (double)rand() / RAND_MAX - 1;
		for (int j = 0;j < l;j ++)
			Pa[i][j] = 2 * (double)rand() / RAND_MAX - 1;
	}
#pragma omp parallel for
	for (int i = 0;i < Ns;i ++)
		for (int j = 0;j < l;j ++)
		{
			V[i][j] = 2 * (double)rand() / RAND_MAX - 1;
			for (int k = 0;k < Nslot;k ++)
				Vt[i][k][j] = 2 * (double)rand() / RAND_MAX - 1;
		}
	return true;
}/*}}}*/

bool save_model(const char *ofn, const uvec &Ca, const uvec &C, const vector<uvec> &Pa, const vector<uvec> &P, const vector<uvec> &V, const vector<vector<uvec> > &Vt)/*{{{*/
{
	ofstream ofs(ofn);
	if (!ofs)
		throw runtime_error("Can not open modelfile for output");
	ofs << V.size() << endl;
	ofs << Ca.size() << endl;
	ofs << C.size() << endl;
	ofs << Vt[0].size() << endl;
	ofs << Ca << endl;
	ofs << C << endl;
	for (int i = 0;i < Pa.size();i ++)
		ofs << Pa[i] << endl;
	for (int i = 0;i < P.size();i ++)
		ofs << P[i] << endl;
	for (int i = 0;i < V.size();i ++)
		ofs << V[i] << endl;
	for (int i = 0;i < Vt.size();i ++)
		for (int j = 0;j < Vt[i].size();j ++)
			ofs << Vt[i][j] << endl;
	ofs.close();
	return true;
}/*}}}*/

Theta load_model(const char *ifn)/*{{{*/
{
	ifstream ifs(ifn);
	if (!ifs)
		throw runtime_error("Could not open modelfile\n");
	size_t Ns, Na, Nt, Nslot;
	ifs >> Ns >> Na >> Nt >> Nslot;
	Theta theta(Ns, Na, Nt, Nslot);
	ifs >> theta.Ca >> theta.C;
	for (int i = 0;i < Na;i ++)
		ifs >> theta.Pa[i];
	for (int i = 0;i < Nt;i ++)
		ifs >> theta.P[i];
	for (int i = 0;i < Ns;i ++)
		ifs >> theta.V[i];
	for (int i = 0;i < Ns;i ++)
		for (int j = 0;j < Nslot;j ++)
			ifs >> theta.Vt[i][j];
	return theta;
}/*}}}*/

int main(int argc, const char *argv[])
{
	//Parsing arguments/*{{{*/
	po::options_description desc("Available options");
	desc.add_options()
		("help,h", "show this help message")
		("iter,it", po::value<unsigned>(&niter)->default_value(20), "set the number of iterations")
		(",l", po::value<unsigned>(&l)->default_value(20), "set the dimension of latent space")
		("nslot", po::value<unsigned>(&Nslot)->default_value(8), "set the number of slots per day")
		("cv,v", po::value<unsigned>(&ncv)->default_value(0), "set the number of folding for cross validation")
		(",w", po::value<unsigned>(&w)->default_value(30 * 60), "set the time window size for short term history (in seconds)")
		("mode,m", po::value<string>(&mode)->default_value("brute"), "set the mode when processing Pst")
		("lambda", po::value<double>(&lambda)->default_value(1e-4), "set the weight decay constant")
		("fraction,f", po::value<double>(&fraction)->default_value(0.75), "set the fraction of training data")
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

	//Load input/*{{{*/
	map<string, unsigned> uids, artids, traids;
	map<unsigned, unsigned> a;
	vector<unsigned> S;
	map<unsigned, vector<example> > D;
	load_tsv(vm["logfile"].as<string>().c_str(), uids, artids, traids, a, S, D);
	map<unsigned, vector<example> > Dtr;
	for (auto const &it : D)
	{
		unsigned s = it.first;
		const vector<example> &Ps = it.second;
		size_t len = std::ceil(Ps.size() * fraction);
		Dtr[s] = vector<example>(Ps.begin(), Ps.begin() + len);
	}
/*}}}*/

	//Initialize parameters/*{{{*/
	unsigned Ns = Dtr.size();
	unsigned Na = artids.size();
	unsigned Nt = traids.size();
	srand(time(NULL));
	cout << "Ns = " << Ns << ", Na = " << Na << ", Nt = " << Nt << endl;
	uvec Ca(Na), C(Nt);
	vector<uvec> Pa(Na, uvec(l)), P(Nt, uvec(l)), V(Ns, uvec(l));
	vector<vector<uvec> > Vt(Ns, vector<uvec>(Nslot, uvec(l)));
	vector<vector<unsigned> > J(Ns);
	init_theta(Ca, C, Pa, P, V, Vt, Na, Nt, Ns);
/*}}}*/

	//print_theta(D, Ca, C, Pa, P, V, Vt);

	//Training iterations/*{{{*/
	for (int k = 0;k < niter;k ++)
	{
		boost::timer::auto_cpu_timer ct("iteration takes %ws\n");
		double eta = 5e-3 / (k + 1);

		sgd(Dtr, a, S, Ca, C, Pa, P, V, Vt, J, eta, mode);
	}
/*}}}*/

	if (vm.count("modelfile"))
		save_model(vm["modelfile"].as<string>().c_str(), Ca, C, Pa, P, V, Vt);
	return 0;
}

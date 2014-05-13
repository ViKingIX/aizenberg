/*
 * koren-pred.cpp
 * Copyright (C) 2014 ViKing <ViKingIX@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <boost/program_options.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "aizenberg.h"

using namespace std;
namespace po = boost::program_options;
namespace ublas = boost::numeric::ublas;

double fraction;	//the fraction of testing examples
string mode;		//the mode to choose pair of testing examples
unsigned w = 60 * 30;	//the size of time window (in seconds)
const string prog_name = "koren-pred";

void print_usage(const po::options_description &desc, const po::positional_options_description &pd)/*{{{*/
{
	cout << "Usage: " << prog_name << " [options] logfile modelfile" << endl << endl;
	cout << desc << endl;
	return;
}/*}}}*/

unsigned select(const vector<unsigned> &S, const set<unsigned> &Suni, const string &mode)
{
	unsigned res;
	auto Suni_it = Suni.cbegin();

	if (mode == "uni")
	{
		advance(Suni_it, rand() % Suni.size());
		res = *(Suni_it);
	}
	else if (mode == "pop")
		res = *(S.cbegin() + rand() % S.size());
	return res;
}

int main(int argc, const char *argv[])
{
	//Argument Parsing/*{{{*/
	po::options_description desc("Available options:");
	desc.add_options()
		("fraction,f", po::value<double>(&fraction)->default_value(0.25), "set the fraction of testing examples")
		//("mode,m", po::value<string>(&mode)->default_value("pop"), "set the mode to pair songs")
		("logfile", po::value<string>()->required(), "path to logfile")
		("modelfile", po::value<string>()->required(), "path to modelfile")
		;
	po::positional_options_description pd;
	pd.add("logfile", 1).add("modelfile", 1);

	po::variables_map vm;
	try
	{
		po::store(po::command_line_parser(argc, argv)
				.options(desc)
				.positional(pd)
				.run(), vm);
		if (vm.count("help"))
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
	
	//load logfile
	map<string, unsigned> uids, artids, traids;
	map<unsigned, unsigned> a;
	vector<unsigned> S;
	map<unsigned, vector<example> > D;
	unsigned Ns, Na, Nt;
	load_tsv(vm["logfile"].as<string>().c_str(), uids, artids, traids, a, S, D, Ns, Na, Nt);
	vector<set<unsigned> > played(uids.size());
	set<unsigned> Suni;
	for (const auto &kv : D)
	{
		unsigned s = kv.first;
		const vector<example> &Ps = kv.second;
		unsigned len = std::ceil(Ps.size() * (1 - fraction));
		for (auto it = Ps.begin();it != Ps.begin() + len;it ++)
			played[s].insert(it->tra);
		for (auto ex : Ps)
			Suni.insert(ex.tra);
	}

	//load model
	Theta theta(vm["modelfile"].as<string>());

	srand(time(NULL));

	//calculate accuracy
	unsigned Nnu = 0, Nnut = 0;	//stats for non played item (uni)
	unsigned Npu = 0, Nput = 0;	//stats for played item (uni)
	unsigned Nnp = 0, Nnpt = 0;	//stats for non played item (pop)
	unsigned Npp = 0, Nppt = 0;	//stats for played item (pop)
	for (const auto &kv : D)
	{
		unsigned s = kv.first;
		const vector<example> &Ps = kv.second;
		unsigned len = std::ceil(Ps.size() * (1 - fraction));
		vector<example> Pstw;
		example ex = *(Ps.begin() + len);
		for (const auto &exj : Ps)
			if (exj.t < ex.t && (ex.t - exj.t) < w)
				Pstw.push_back(exj);
		for (auto it = Ps.begin() + len;it != Ps.end();it ++)
		{
			unsigned ai = it->art, i = it->tra;
			ublas::vector<double> qsum = ublas::zero_vector<double>(theta.l);
			for (auto it2 = Pstw.begin();it2 != Pstw.end();)
				if (it->t - it2->t < w)
				{
					qsum += theta.Pa[it2->art] + theta.P[it2->tra];
					it2 ++;
				}
				else
					it2 = Pstw.erase(it2);
			double coeff = 0;
			if (Pstw.size() > 0)
				coeff = 1. / sqrt(Pstw.size());
			double bi = theta.Ca[ai] + theta.C[i];
			ublas::vector<double> qi = theta.Pa[ai] + theta.P[i];
			unsigned slot = it->t % 86400 / 60 / 60 / theta.Nslot;
			ublas::vector<double> vterm = theta.V[s] + theta.Vt[s][slot];
			double expri = exp(bi + inner_prod(qi, vterm));

			unsigned ju = select(S, Suni, "uni"), aju = a[ju];
			double bju = theta.Ca[aju] + theta.C[ju];
			ublas::vector<double> qju = theta.Pa[aju] + theta.P[ju];
			double exprju = exp(bju + inner_prod(qju, vterm));

			unsigned jp = select(S, Suni, "pop"), ajp = a[jp];
			double bjp = theta.Ca[ajp] + theta.C[jp];
			ublas::vector<double> qjp = theta.Pa[ajp] + theta.P[jp];
			double exprjp = exp(bjp + inner_prod(qjp, vterm));

			if (played[s].count(ex.tra))
			{
				if (expri > exprju)
					Nput++;
				if (expri > exprjp)
					Nppt++;
				Npu++;
				Npp++;
			}
			else
			{
				if (expri > exprju)
					Nnut++;
				if (expri > exprjp)
					Nnpt++;
				Nnu++;
				Nnp++;
			}
			Pstw.push_back(ex);
		}
	}
	cout << "NonRepeat-vs-Uni: " << Nnut << "/" << Nnu << " = " << (double)Nnut / Nnu << endl;
	cout << "Played-vs-Uni: " << Nput << "/" << Npu << " = " << (double)Nput / Npu << endl;
	cout << "NonRepeat-vs-Pop: " << Nnpt << "/" << Nnp << " = " << (double)Nnpt / Nnp << endl;
	cout << "Played-vs-Pop: " << Nppt << "/" << Npp << " = " << (double)Nppt / Npp << endl;
	return 0;
}

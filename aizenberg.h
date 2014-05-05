/*
 * aizenberg.h
 * Copyright (C) 2014 ViKing <ViKingIX@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef AIZENBERG_H
#define AIZENBERG_H
namespace ublas = boost::numeric::ublas;

struct example /*{{{*/
{
	example(unsigned s, unsigned i, unsigned a, time_t t) : s(s), tra(i), art(a), t(t) {}
	unsigned s;
	unsigned tra;
	unsigned art;
	time_t t;
};/*}}}*/

struct Theta/*{{{*/
{
	Theta(size_t Ns, size_t Na, size_t Nt, size_t Nslot, size_t l = 0) : Ns(Ns), Na(Na), Nt(Nt), Nslot(Nslot)/*{{{*/
	{
		Ca.resize(l);
		C.resize(l);
		Pa.resize(Na);
#pragma omp parallel for
		for (int i = 0;i < Na;i ++)
			Pa[i].resize(l);
		P.resize(Nt);
#pragma omp parallel for
		for (int i = 0;i < Nt;i ++)
			P[i].resize(l);
		V.resize(Ns);
		Vt.resize(Ns);
#pragma omp parallel for
		for (int i = 0;i < Ns;i ++)
		{
			V[i].resize(l);
			Vt[i].resize(Nslot);
			for (int j = 0;j < Nslot;j ++)
				Vt[i][j].resize(l);
		}
	}/*}}}*/
	Theta(const std::string &fn)/*{{{*/
	{
		loads(fn);
	}/*}}}*/
	void init()/*{{{*/
	{
#pragma omp parallel for
		for (int i = 0;i < Nt;i ++)
		{
			C[i] = 2 * (double)rand() / RAND_MAX - 1;
			for (int j = 0;j < P[i].size();j ++)
				P[i][j] = 2 * (double)rand() / RAND_MAX - 1;
		}
#pragma omp parallel for
		for (int i = 0;i < Na;i ++)
		{
			Ca[i] = 2 * (double)rand() / RAND_MAX - 1;
			for (int j = 0;j < Pa[i].size();j ++)
				Pa[i][j] = 2 * (double)rand() / RAND_MAX - 1;
		}
#pragma omp parallel for
		for (int i = 0;i < Ns;i ++)
			for (int j = 0;j < V[i].size();j ++)
			{
				V[i][j] = 2 * (double)rand() / RAND_MAX - 1;
				for (int k = 0;k < Nslot;k ++)
					Vt[i][k][j] = 2 * (double)rand() / RAND_MAX - 1;
			}
		return;
	}/*}}}*/
	void loads(const std::string &fn)/*{{{*/
	{
		std::ifstream ifs(fn.c_str(), std::ios::in);

		ifs >> Ns >> Na >> Nt >> Nslot;
		V.resize(Ns);
		Vt.resize(Ns);
		for (size_t i = 0;i < Ns;i ++)
			Vt[i].resize(Nslot);
		Pa.resize(Na);
		P.resize(Nt);

		ifs >> Ca >> C;
		for (size_t i = 0;i < Na;i ++)
			ifs >> Pa[i];
		for (size_t i = 0;i < Nt;i ++)
			ifs >> P[i];
		for (size_t i = 0;i < Ns;i ++)
			ifs >> V[i];
		for (size_t i = 0;i < Ns;i ++)
			for (size_t j = 0;j < Nslot;j ++)
				ifs >> V[i][j];
		return;
	}/*}}}*/
	void saves(const std::string &fn)/*{{{*/
	{
		std::ofstream ofs(fn.c_str(), std::ios::out);

		ofs << Ns << endl;
		ofs << Na << endl;
		ofs << Nt << endl;
		ofs << Nslot << endl;
		ofs << Ca << endl;
		ofs << C << endl;
		for (size_t i = 0;i < Na;i ++)
			ofs << Pa[i] << endl;
		for (size_t i = 0;i < Nt;i ++)
			ofs << P[i] << endl;
		for (size_t i = 0;i < Ns;i ++)
			ofs << V[i] << endl;
		for (size_t i = 0;i < Ns;i ++)
			for (size_t j = 0;j < Nslot;j ++)
				ofs << Vt[i][j] << endl;
		return;
	}/*}}}*/
	size_t Ns, Na, Nt, Nslot;
	ublas::vector<double> Ca, C;
	std::vector<ublas::vector<double> > Pa, P, V;
	std::vector<std::vector<ublas::vector<double> > > Vt;
};/*}}}*/

#endif /* !AIZENBERG_H */

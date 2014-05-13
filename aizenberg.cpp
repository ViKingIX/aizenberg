/*
 * aizenberg.cpp
 * Copyright (C) 2014 ViKing <ViKingIX@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include <iostream>
#include <map>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <boost/algorithm/string.hpp>
#include <boost/date_time.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/timer/timer.hpp>
#include "aizenberg.h"

using namespace std;
namespace ublas = boost::numeric::ublas;
namespace bt = boost::posix_time;

Theta::Theta(unsigned Ns, unsigned Na, unsigned Nt, unsigned Nslot, unsigned l) : Ns(Ns), Na(Na), Nt(Nt), Nslot(Nslot), l(l)/*{{{*/
{
	Ca.resize(Na);
	C.resize(Nt);
	Pa.resize(Na);
	for (int i = 0;i < Na;i ++)
		Pa[i].resize(l);
	P.resize(Nt);
	for (int i = 0;i < Nt;i ++)
		P[i].resize(l);
	V.resize(Ns);
	Vt.resize(Ns);
	for (int i = 0;i < Ns;i ++)
	{
		V[i].resize(l);
		Vt[i].resize(Nslot);
		for (int j = 0;j < Nslot;j ++)
			Vt[i][j].resize(l);
	}
}/*}}}*/
void Theta::init()/*{{{*/
{
	srand(time(NULL));
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
	return;
}/*}}}*/
void Theta::loads(const std::string &fn)/*{{{*/
{
	boost::timer::auto_cpu_timer ct("load theta takes %ws\n");
	std::ifstream ifs(fn.c_str(), std::ios::in);

	ifs >> Ns >> Na >> Nt >> Nslot >> l;
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
			ifs >> Vt[i][j];
	return;
}/*}}}*/
void Theta::saves(const std::string &fn) const/*{{{*/
{
	std::ofstream ofs(fn.c_str(), std::ios::out);

	ofs << Ns << std::endl;
	ofs << Na << std::endl;
	ofs << Nt << std::endl;
	ofs << Nslot << std::endl;
	ofs << l << std::endl;
	ofs << Ca << std::endl;
	ofs << C << std::endl;
	for (size_t i = 0;i < Na;i ++)
		ofs << Pa[i] << std::endl;
	for (size_t i = 0;i < Nt;i ++)
		ofs << P[i] << std::endl;
	for (size_t i = 0;i < Ns;i ++)
		ofs << V[i] << std::endl;
	for (size_t i = 0;i < Ns;i ++)
		for (size_t j = 0;j < Nslot;j ++)
			ofs << Vt[i][j] << std::endl;
	return;
}/*}}}*/
void Theta::dump() const/*{{{*/
{
	cout << "Ns = " << Ns << ", " << "Na = " << Na << ", " << "Nt = " << Nt << endl;
	cout << Ca << endl;
	cout << C << endl;
	for (int i = 0;i < Pa.size();i ++)
		cout << Pa[i] << endl;
	for (int i = 0;i < P.size();i ++)
		cout << P[i] << endl;
	for (int i = 0;i < V.size();i ++)
		cout << V[i] << endl;
	for (int i = 0;i < Vt.size();i ++)
		for (int j = 0;j < Nslot;j ++)
			cout << Vt[i][j] << endl;
	return;
}/*}}}*/

bool load_tsv(const char *logfilefn, map<string, unsigned> &uids, map<string, unsigned> &artids, map<string, unsigned> &traids, map<unsigned, unsigned> &a, vector<unsigned> &S, map<unsigned, vector<example> > &D, unsigned &Ns, unsigned &Na, unsigned &Nt)/*{{{*/
{
	boost::timer::auto_cpu_timer ct("load_tsv takes %ws\n");
	const locale datetime_fmt(locale::classic(), new bt::time_input_facet("%Y-%m-%dT%H:%M:%SZ"));
	ifstream logfile(logfilefn);
	if (!logfile)
		throw runtime_error("File not found!");
	string line;
	while (getline(logfile, line))
	{
		vector<string> fields;

		boost::split(fields, line, boost::algorithm::is_any_of("\t"), boost::token_compress_on);
		if (fields.size() < 6)
		{
			cerr << "bad format for log: " << line << endl;
			continue;
		}

		//userid, timestamp, artid, artname, traid, traname
		string userid = fields[0];
		string timestr = fields[1];
		string artid = fields[2];
		string artname = fields[3];
		string traid = fields[4];
		string traname = fields[5];

		if (!uids.count(userid))
			uids[userid] = uids.size() - 1;
		unsigned s = uids[userid];

		if (!artids.count(artid))
			artids[artid] = artids.size() - 1;
		unsigned ai = artids[artid];

		if (!traids.count(traid))
			traids[traid] = traids.size() - 1;
		unsigned i = traids[traid];
		S.push_back(i);
		a[i] = ai;

		bt::ptime pt;
		istringstream is(timestr);
		is.imbue(datetime_fmt);
		is >> pt;
		if (pt == bt::ptime())
		{
			cerr << "bad time format: " << timestr << endl;
			continue;
		}
		time_t t = (pt - bt::ptime(boost::gregorian::date(1970, 1, 1))).ticks() / bt::time_duration::rep_type::ticks_per_second;

		D[s].push_back(example(s, i, ai, t));
	}
	logfile.close();
	Ns = uids.size();
	Na = artids.size();
	Nt = traids.size();
	return true;
}/*}}}*/

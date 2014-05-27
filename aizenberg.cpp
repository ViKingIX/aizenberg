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

Theta::Theta(const std::map<string, unsigned> &uids, const std::map<string, unsigned> &artids, const std::map<string, unsigned> &traids, unsigned Nslot, unsigned l) : Nslot(Nslot), l(l)/*{{{*/
{
	unsigned i;
	ublas::zero_vector<double> z(l);
	for (const auto &kv : uids)
	{
		i = kv.second;
		V[i] = z;
		Vt[i].resize(Nslot);
		for (int j = 0;j < Nslot;j ++)
			Vt[i][j] = z;
	}
	for (const auto &kv : artids)
	{
		i = kv.second;
		Ca[i] = 0;
		Pa[i] = z;
	}
	for (const auto &kv : traids)
	{
		i = kv.second;
		C[i] = 0;
		P[i] = z;
	}
}/*}}}*/
void Theta::init()/*{{{*/
{
	srand(time(NULL));
	for (auto &kv : C)
	{
		unsigned i = kv.first;
		C[i] = 2 * (double)rand() / RAND_MAX - 1;
		for (int j = 0;j < l;j ++)
			P[i][j] = 2 * (double)rand() / RAND_MAX - 1;
	}
	for (auto &kv : Ca)
	{
		unsigned i = kv.first;
		Ca[i] = 2 * (double)rand() / RAND_MAX - 1;
		for (int j = 0;j < l;j ++)
			Pa[i][j] = 2 * (double)rand() / RAND_MAX - 1;
	}
	for (auto &kv : V)
	{
		unsigned i = kv.first;
		for (int j = 0;j < l;j ++)
		{
			V[i][j] = 2 * (double)rand() / RAND_MAX - 1;
			for (int k = 0;k < Nslot;k ++)
				Vt[i][k][j] = 2 * (double)rand() / RAND_MAX - 1;
		}
	}
	return;
}/*}}}*/
void Theta::loads(const std::string &fn)/*{{{*/
{
	boost::timer::auto_cpu_timer ct("load theta takes %ws\n");
	std::ifstream ifs(fn.c_str(), std::ios::in);
	unsigned Ns, Na, Nt, x;

	ifs >> Ns >> Na >> Nt >> Nslot >> l;

	for (int i = 0;i < Na;i ++)
	{
		ifs >> x;
		ifs >> Ca[x] >> Pa[x];
	}
	for (int i = 0;i < Nt;i ++)
	{
		ifs >> x;
		ifs >> C[x] >> P[x];
	}
	for (int i = 0;i < Ns;i ++)
	{
		ifs >> x;
		ifs >> V[x];
		Vt[x].resize(Nslot);
		for (int j = 0;j < Nslot;j ++)
			ifs >> Vt[x][j];
	}
	return;
}/*}}}*/
void Theta::saves(const string &fn) const/*{{{*/
{
	ofstream ofs(fn.c_str(), ios::out);

	ofs << V.size() << " " << Ca.size() << " " << C.size() << " " << Nslot << " " << l << endl;
	for (auto &kv : Ca)
	{
		unsigned a = kv.first;
		ofs << a << " " << Ca.find(a)->second << " " << Pa.find(a)->second << endl;
	}
	for (auto &kv : C)
	{
		unsigned i = kv.first;
		ofs << i << " " << C.find(i)->second << " " << P.find(i)->second << endl;
	}
	for (auto &kv : V)
	{
		unsigned s = kv.first;
		ofs << s << " " << V.find(s)->second;
		for (int j = 0;j < Nslot;j ++)
			ofs << " " << Vt.find(s)->second[j];
		ofs << endl;
	}
	return;
}/*}}}*/
void Theta::dump() const/*{{{*/
{
	cout << "Ns = " << V.size() << ", " << "Na = " << Ca.size() << ", " << "Nt = " << C.size() << endl;
#if 0
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
#endif
	return;
}/*}}}*/

bool load_tsv(const char *logfilefn, map<string, unsigned> &uids, map<string, unsigned> &artids, map<string, unsigned> &traids, map<unsigned, unsigned> &a, map<unsigned, vector<example> > &D)/*{{{*/
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
	return true;
}/*}}}*/

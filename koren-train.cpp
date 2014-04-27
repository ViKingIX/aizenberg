/*
 * koren-train.cpp
 * Copyright (C) 2014 ViKing <ViKingIX@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include <iostream>
#include <cstring>
#include <fstream>
#include <exception>
#include <vector>
#include <map>
#include <ctime>
#include <cstdlib>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/date_time.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/timer/timer.hpp>

using namespace std;
namespace ublas = boost::numeric::ublas;
typedef ublas::vector<double> uvec;
typedef ublas::zero_vector<double> zvec;
namespace po = boost::program_options;
namespace bt = boost::posix_time;

unsigned niter;		//number of iterations
unsigned ncv;		//number of foldings
unsigned l;		//number of dimensions
unsigned Nslot;		//number of slots per day
unsigned w;		//size of time window (in seconds)
unsigned nfields = 6;	//number of fields to be recognized as data
const unsigned SEC_PER_DAY = 24 * 60 * 60, SEC_PER_HOUR = 60 * 60;
const string prog_name = "koren-train";
const locale datetime_fmt(locale::classic(), new bt::time_input_facet("%Y-%m-%dT%H:%M:%SZ"));

void print_usage(const po::options_description &desc, const po::positional_options_description &pd)/*{{{*/
{
	cout << "Usage: " << prog_name << " [options] logfile [modelfile]" << endl << endl;
	cout << desc << endl;
	return;
}/*}}}*/

class example /*{{{*/
{
	public:
		example(unsigned s, unsigned i, unsigned a, time_t t) : s(s), tra(i), art(a), t(t) {}
		unsigned s;
		unsigned tra;
		unsigned art;
		time_t t;
};/*}}}*/

#if 0
template <class T1, class T2>
vec vec_add(const T1 &v1, const T2 &v2)/*{{{*/
{
	vec v(v1.size());
	for (int i = 0;i < v1.size();i ++)
		v[i] = v1[i] + v2[i];
	return v;
}/*}}}*/

double vec_dot(const vec &v1, const vec &v2)/*{{{*/
{
	if (v1.size() != v2.size())
		return 0;
	double dot = 0;
	for (int i = 0;i < v1.size();i ++)
		dot += v1[i] * v2[i];
	return dot;
}/*}}}*/

vec vec_mul(double a, const vec &v1)/*{{{*/
{
	vec v(v1);
	for (int i = 0;i < v.size();i ++)
		v[i] *= a;
	return v;
}/*}}}*/
#endif

inline double clip(double x, double low, double high)/*{{{*/
{
	return min(max(x, high), low);
}/*}}}*/

inline uvec vclip(uvec vin, double low, double high)/*{{{*/
{
	uvec v(vin);
	for (int i = 0;i < v.size();i ++)
		v[i] = min(max(v[i], high), low);
	return v;
}/*}}}*/

#if 0
bool update(const map<int, vector<example> > &D, const vector<int> &S, matrix<double> &C, matrix<double> &Ca, matrix<double> &P, matrix<double> &Pa, matrix<double> &V, vector<matrix<double> > &Vt, double eta)/*{{{*/
{
	bool converge = false;
	static vector<int> random_s(D.size());
	
	for (int i = 0;i < random_s.size();i ++)
		random_s[i] = i;
	random_shuffle(random_s.begin(), random_s.end());
	for (int x = 0;x < random_s.size();x ++)
	{
		int s = random_s[x];
		vector<example> Ps = D.at(s), Pst, Pstw;

		for (int i = 0;i < Ps.size();i ++)
		{
			example exi = Ps[i];
			vec qsum(l, 0);

			//remove j not in (t - w, t] and calculate qsum
			for (auto it = Pstw.begin();it != Pstw.end();)
				if (it->t < exi.t && exi.t - it->t >= w)
					it = Pstw.erase(it);
				else
				{
					qsum = vec_add(qsum, vec_add(row(Pa, exi.art), row(P, exi.tra)));
					it ++;
				}
			double coeff = 1. / sqrt(Pstw.size());
			for (int x = 0;x < l;x ++)
				qsum[i] *= coeff;

			double bi = Ca(exi.art, 0) + C(exi.tra, 0);
			vec qi = vec_add(row(Pa, exi.art), row(P, exi.tra));

			int slot = exi.t % 86400 / 3600;
			vec vterm = vec_add(row(V, s), vec_add(row(Vt[s], slot), qsum));
			map<int, double> dCa, dC;
			map<int, uvec> dPa, dP;
			uvec dV(l), dVt(l);
			dCa[exi.art] = dC[exi.tra] = eta;
			dPa[exi.art] = dP[exi.tra] = eta * uvec(vterm);
			dV = dVt = eta * uvec(qi);
			Pstw.push_back(exi);

			//rsj;t part
			//brute force
			vector<example> Pst;
			for (int x = 0;x < Ps.size();x ++)
			{
				const example &exj = Ps[x];
				if (exj.t % 86400 / 3600 == exi.t % 86400 / 3600 && exj.t != exi.t)
					Pst.push_back(exj);
			}

			map<int, double> expr;
			double denom = 0;
			for (int x = 0;x < Pst.size();x ++)
			{
				const example &exj = Pst[x];
				double bj = Ca(exj.art, 0) + C(exj.tra, 0);
				vec qj = vec_add(row(Pa, exj.art), row(P, exj.tra));
				if (!expr.count(exj.tra))
					expr[exj.tra] = 0;
				expr[exj.tra] += exp(bj + vec_dot(qj, vterm));
				denom += expr[exj.tra];
			}
			for (int x = 0;x < Pst.size();x ++)
			{
				const example &exj = Pst[x];
				int aj = exj.art, j = exj.tra;
				vec qj = vec_add(row(Pa, exj.art), row(P, exj.tra));
				double coeff2 = -eta * expr[exj.tra] / denom;
				if (!dPa.count(aj))
					dPa[aj] = vec(l, 0);
				if (!dCa.count(aj))
					dCa[aj] = 0;
				dCa[aj] += coeff2;
				dC[j] += coeff2;
				dPa[aj] += vec_mul(coeff2, vterm);
				dP[j] += vec_mul(coeff2, vterm);
				dV += vec_mul(coeff2, qj);
				dVt += vec_mul(coeff2, qj);
				for (int k = 0;k < Pstw.size();k ++)
				{
					const example &exk = Pstw[k];

					if (!dP.count(exk.tra))
						dP[exk.tra] = vec(l, 0);
				}
			}
#if 0
			for (auto it = dCa.begin();it != dCa.end();it ++)
				Ca[it->first] = clip(Ca[it->first] + it->second, -1, 1);
			for (auto it = dC.begin();it != dC.end();it ++)
				C[it->first] = clip(C[it->first] + it->second, -1, 1);
			for (auto it = dPa.begin();it != dPa.end();it ++)
				Pa[it->first] = vec_clip(vec_add(Pa[it->first], it->second), -1, 1);
			for (auto it = dP.begin();it != dP.end();it ++)
				P[it->first] = vec_clip(vec_add(P[it->first], it->second), -1, 1);
			V[s] = vec_clip(vec_add(V[s], dV), -1, 1);
			Vt[s][slot] = vec_clip(vec_add(Vt[s][slot], dVt), -1, 1);
#endif
		}
	}
	return converge;
}/*}}}*/
#endif
bool sgd(const map<unsigned, vector<example> > &D, const vector<unsigned> &S, uvec &Ca, uvec &C, vector<uvec> &Pa, vector<uvec> &P, vector<uvec> &V, vector<vector<uvec> > &Vt, double eta, string mode = "brute")/*{{{*/
{
	bool converge = false;
	static vector<unsigned> random_s;
	
	if (!random_s.size())
	{
		random_s.resize(D.size());
		for (int i = 0;i < random_s.size();i ++)
			random_s[i] = i;
	}
	random_shuffle(random_s.begin(), random_s.end());
	for (int x = 0;x < random_s.size();x ++)
	{
		unsigned s = random_s[x];
		cout << "D.size() = " << D.size() << ", s = " << s << endl;
		vector<example> Ps = D.find(s)->second, Pst, Pstw;

		for (int idxi = 0;idxi < Ps.size();idxi ++)
		{
			example exi = Ps[idxi];
			uvec qsum = zvec(l);

			//remove j not in (t - w, t] and calculate qsum
			for (vector<example>::iterator it = Pstw.begin();it != Pstw.end();)
				if (it->t < exi.t && exi.t - it->t >= w)
					it = Pstw.erase(it);
				else
				{
					qsum += Pa[exi.art] + P[exi.tra];
					it ++;
				}
			Pstw.push_back(exi);

			double coeff = 1 / sqrt(Pstw.size());
			unsigned ai = exi.art, i = exi.tra;
			double bi = Ca[ai] + C[i];
			uvec qi = Pa[ai] + P[i];
			unsigned slot = exi.t % SEC_PER_DAY / SEC_PER_HOUR / Nslot;
			uvec vterm = V[s] + Vt[s][slot] + coeff * qsum;

			map<unsigned, double> dCa, dC;
			map<unsigned, uvec> dPa, dP;
			uvec dV, dVt;
			dCa[ai] = dC[i] = eta;
			dPa[ai] = dP[i] = eta * vterm;
			dV = dVt = eta * qi;

			//rsj;t part
			if (mode == "brute")/*{{{*/
			{
				boost::timer::auto_cpu_timer ct("brute force costs %ws\n");
				vector<example> Pst;
				for (int x = 0;x < Ps.size();x ++)
				{
					const example &exj = Ps[x];
					if (exj.t % 86400 / 3600 == exi.t % 86400 / 3600 && exj.t != exi.t)
						Pst.push_back(exj);
				}

				map<int, double> expr;
				double denom = 0;
				for (int x = 0;x < Pst.size();x ++)
				{
					const example &exj = Pst[x];
					unsigned aj = exj.art, j = exj.tra;
					double bj = Ca[aj] + C[j];
					uvec qj = Pa[aj] + P[j];
					if (!expr.count(j))
						expr[j] = 0;
					double exprj = exp(bj + inner_prod(qj, vterm));
					expr[j] += exprj;
					denom += exprj;
				}
				for (int x = 0;x < Pst.size();x ++)
				{
					const example &exj = Pst[x];
					unsigned aj = exj.art, j = exj.tra;
					uvec qj = Pa[aj] + P[j];
					double coeff2 = -eta * expr[j] / denom;
					if (!dCa.count(aj))
						dCa[aj] = 0;
					if (!dC.count(j))
						dC[j] = 0;
					if (!dPa.count(aj))
						dPa[aj] = zvec(l);
					if (!dP.count(j))
						dPa[j] = zvec(l);
					dCa[aj] += coeff2;
					dC[j] += coeff2;
					dPa[aj] += coeff2 * vterm;
					dP[j] += coeff2 * vterm;
					dV += coeff2 * qj;
					dVt += coeff2 * qj;
					for (int k = 0;k < Pstw.size();k ++)
					{
						const example &exk = Pstw[k];

						if (!dP.count(exk.tra))
							dP[exk.tra] = zvec(l);
					}
				}
			}/*}}}*/
			else if (mode == "imp")/*{{{*/
			{
				boost::timer::auto_cpu_timer ct("importance sampling costs %ws\n");
			}/*}}}*/

#if 0
			//update/*{{{*/
			for (map<unsigned, double>::iterator it = dCa.begin();it != dCa.end();it ++)
				Ca[it->first] = clip(Ca[it->first] + it->second, -1, 1);
			for (map<unsigned, double>::iterator it = dC.begin();it != dC.end();it ++)
				C[it->first] = clip(C[it->first] + it->second, -1, 1);
			cout << "updated C\n";
			cout << "Pa.size() = " << Pa.size() << endl;
			for (map<unsigned, uvec>::iterator it = dPa.begin();it != dPa.end();it ++)
			{
				cout << "update art " << it->first << endl;
				Pa[it->first] = vclip(Pa[it->first] + it->second, -1, 1);
			}
			cout << "updated Pa\n";
			cout << "P.size() = " << P.size() << endl;
			for (map<unsigned, uvec>::iterator it = dP.begin();it != dP.end();it ++)
			{
				cout << "update tra " << it->first << endl;
				P[it->first] = vclip(P[it->first] + it->second, -1, 1);
			}
			cout << "updated P\n";
			V[s] = vclip(V[s] + dV, -1, 1);
			cout << "updated V\n";
			Vt[s][slot] = vclip(Vt[s][slot] + dVt, -1, 1);
			cout << "updated Vt\n";
			/*}}}*/
			cout << "update finished\n";
#endif
		}
	}
	return converge;
}/*}}}*/

bool load_input(const char *logfilefn, map<string, unsigned> &uids, map<string, unsigned> &artids, map<string, unsigned> &traids, map<unsigned, unsigned> &a, vector<unsigned> &S, map<unsigned, vector<example> > &D)/*{{{*/
{
	boost::timer::auto_cpu_timer ct("load_input costs %ws\n");
	ifstream logfile(logfilefn);
	do
	{
		string line;
		vector<string> fields;

		getline(logfile, line);
		boost::split(fields, line, boost::algorithm::is_space(), boost::token_compress_on);
		if (fields.size() < nfields)
			continue;

		//userid, timestamp, artid, artname, traid, traname
		string userid = fields[0];
		string timestr = fields[1];
		string artid = fields[2];
		string artname = fields[3];
		string traid = fields[4];
		string traname = fields[5];

		if (!uids.count(userid))
			uids[userid] = uids.size();
		unsigned s = uids[userid];

		if (!artids.count(artid))
			artids[artid] = artids.size();
		unsigned ai = artids[artid];

		if (!traids.count(traid))
			traids[traid] = traids.size();
		unsigned i = traids[traid];
		S.push_back(i);
		a[i] = ai;

		bt::ptime pt;
		istringstream is(timestr);
		is.imbue(datetime_fmt);
		is >> pt;
		if (pt == bt::ptime())
		{
			cerr << "bad time format" << endl;
			continue;
		}
		time_t t = (pt - bt::ptime(boost::gregorian::date(1970, 1, 1))).ticks() / bt::time_duration::rep_type::ticks_per_second;

		D[s].push_back(example(s, i, ai, t));
	} while (!logfile.eof());
	return true;
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
				Vt[i][k][l] = 2 * (double)rand() / RAND_MAX - 1;
		}
	return true;
}/*}}}*/

void print_theta(const map<unsigned, vector<example> > &D, const uvec &Ca, const uvec &C, const vector<uvec> &Pa, const vector<uvec> &P, const vector<uvec> &V, const vector<vector<uvec> > &Vt)
{
	for (map<unsigned, vector<example> >::const_iterator it = D.begin();it != D.end();it ++)
		for (int i = 0;i < it->second.size();i ++)
			cout << "(" << it->second[i].s << ", " << it->second[i].tra << ", " << it->second[i].t << ")\n";
	for (int i = 0;i < Ca.size();i ++)
		cout << Ca[i] << " ";
	cout << endl;
	for (int i = 0;i < C.size();i ++)
		cout << C[i] << " ";
	cout << endl;
	cout << "Pa.size() = " << Pa.size() << endl;
	for (int i = 0;i < Pa.size();i ++)
		cout << Pa[i] << endl;
	cout << endl;
	cout << "P.size() = " << P.size() << endl;
	for (int i = 0;i < P.size();i ++)
		cout << P[i] << endl;
	cout << "V.size() = " << V.size() << endl;
	for (int i = 0;i < V.size();i ++)
		cout << V[i] << endl;
	return;
}

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
	load_input(vm["logfile"].as<string>().c_str(), uids, artids, traids, a, S, D);
/*}}}*/

	//Initialize parameters/*{{{*/
	unsigned Ns = D.size();
	unsigned Na = artids.size();
	unsigned Nt = traids.size();
	srand(time(NULL));
	cout << "Ns = " << Ns << ", Na = " << Na << ", Nt = " << Nt << endl;
#if 0
	matrix<double> C(Nt, 1), Ca(Na, 1), P(Nt, l), Pa(Na, l), V(Ns, l);
	vector<matrix<double> > Vt(Ns, matrix<double>(Nslot, l));
#pragma omp parallel for collapse(2)
	for (int i = 0;i < Nt;i ++)
	{
		C(i, 0) = 2 * double(rand()) / RAND_MAX - 1;
		for (int j = 0;j < l;j ++)
			P(i, j) = 2 * double(rand()) / RAND_MAX - 1;
	}
#pragma omp parallel for collapse(2)
	for (int i = 0;i < Na;i ++)
	{
		Ca(i, 0) = 2 * double(rand()) / RAND_MAX - 1;
		for (int j = 0;j < l;j ++)
			Pa(i, j) = 2 * double(rand()) / RAND_MAX - 1;
	}
#pragma omp parallel for collapse(3)
	for (int i = 0;i < Ns;i ++)
		for (int j = 0;j < l;j ++)
		{
			V(i, j) = 2 * double(rand()) / RAND_MAX - 1;
			for (int k = 0;k < Nslot;k ++)
				Vt[i](k, l) = 2 * double(rand()) / RAND_MAX - 1;
		}
#endif
	uvec Ca(Na), C(Nt);
	vector<uvec> Pa(Na, uvec(l)), P(Nt, uvec(l)), V(Ns, uvec(l));
	vector<vector<uvec> > Vt(Ns, vector<uvec>(Nslot, uvec(l)));
	init_theta(Ca, C, Pa, P, V, Vt, Na, Nt, Ns);
/*}}}*/

	print_theta(D, Ca, C, Pa, P, V, Vt);

	//Training iterations/*{{{*/
	for (int k = 0;k < niter;k ++)
	{
		boost::timer::auto_cpu_timer ct("iteration takes %ws\n");
		double eta = 0.005 / (k + 1);

		sgd(D, S, Ca, C, Pa, P, V, Vt, eta);
	}
/*}}}*/
	return 0;
}

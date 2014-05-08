/*
 * aizenberg.h
 * Copyright (C) 2014 ViKing <ViKingIX@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef AIZENBERG_H
#define AIZENBERG_H

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
	Theta(unsigned Ns, unsigned Na, unsigned Nt, unsigned Nslot, unsigned l);
	Theta(const std::string &fn) {loads(fn);}
	void init();
	void loads(const std::string &fn);
	void saves(const std::string &fn) const;
	void dump() const;

	unsigned Ns, Na, Nt, Nslot, l;
	boost::numeric::ublas::vector<double> Ca, C;
	std::vector<boost::numeric::ublas::vector<double> > Pa, P, V;
	std::vector<std::vector<boost::numeric::ublas::vector<double> > > Vt;
};/*}}}*/

bool load_tsv(const char *logfilefn, std::map<std::string, unsigned> &uids, std::map<std::string, unsigned> &artids, std::map<std::string, unsigned> &traids, std::map<unsigned, unsigned> &a, std::vector<unsigned> &S, std::map<unsigned, std::vector<example> > &D);

inline bool cmp_ts(time_t t1, time_t t2)/*{{{*/
{
	return t1 % 86400 / 60 == t2 % 86400 / 60;
}/*}}}*/

#endif /* !AIZENBERG_H */

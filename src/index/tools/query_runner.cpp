/**
 * @file query_runner.cpp
 * @author Pranav
 */

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

#include "meta/util/time.h"
#include "meta/util/printing.h"
#include "meta/corpus/document.h"
#include "meta/index/inverted_index.h"
#include "meta/index/eval/ir_eval.h"
#include "meta/index/ranker/ranker_factory.h"
#include "meta/parser/analyzers/tree_analyzer.h"
#include "meta/sequence/analyzers/ngram_pos_analyzer.h"
#include "meta/index/score_data.h"
#include "meta/index/ranker/okapi_bm25.h"
#include "meta/index/ranker/jelinek_mercer.h"
#include "meta/index/ranker/dirichlet_prior.h"
#include "meta/index/ranker/pivoted_length.h"
#include "meta/math/fastapprox.h"

using namespace meta;

std::vector < std::vector <std::string> > vec_dict, query_dict;
std::unordered_map <std::string, int> pointer_dict;

template <class Index, class SearchResult>
void print_results(const Index& idx, const SearchResult& result, uint64_t result_num)
{
	std::string path{idx->doc_path(result.d_id)};
	auto output = printing::make_bold(std::to_string(result_num) + ". " + path) + " (score = " + std::to_string(result.score) + ", docid = " + std::to_string(result.d_id) + ")";
	std::cout << output << std::endl;
	auto mdata = idx->metadata(result.d_id);
	if (auto content = mdata.template get<std::string>("content"))
	{
		auto len = std::min(std::string::size_type{77}, content->size());
		std::cout << content->substr(0, len) << "..." << std::endl << std::endl;
	}
}

// PL2 ALG

class pl2_ranker : public index::ranker {
	private:
	float c_ = 7;
	float lambda_ = 0.1;

	public:
	const static util::string_view id;
	pl2_ranker();
	pl2_ranker(float c, float lambda);
	pl2_ranker(std::istream& in);
	void set_param(float c, float lambda)
	{
		c_ = c;
		lambda_ = lambda;
	};
	float score_one(const index::score_data&) override;
	void save(std::ostream& out) const override;
};

const util::string_view pl2_ranker::id = "pl2";
pl2_ranker::pl2_ranker() {}
pl2_ranker::pl2_ranker(float c, float lambda) : c_{c}, lambda_{lambda} {}
pl2_ranker::pl2_ranker(std::istream& in) : c_{io::packed::read<float>(in)}, lambda_{io::packed::read<float>(in)} {}

void pl2_ranker::save(std::ostream& out) const
{
	io::packed::write(out, id);
	io::packed::write(out, c_);
	io::packed::write(out, lambda_);
}

float pl2_ranker::score_one(const index::score_data& sd)
{
	float doc_len = (float)sd.idx.doc_size(sd.d_id);  // Current document length
	float avg_dl = sd.avg_dl;	  // Average document length in the corpus
	float tf = (float)sd.doc_term_count;  // Raw term count in the document
	float pi = 3.14f;			   // Use this for pi - Do NOT use other values
	float lambda = (float)sd.num_docs / (float)sd.corpus_term_count;	  // pl2's parameter
	float c = c_;				  // pl2's parameter
	float tfn = tf * log2(1.0 + c * (avg_dl / doc_len));
	float score = (1.0 / (tfn + 1.0)) * (tfn * log2(tfn / lambda) + (lambda + (1.0 / (12.0 * tfn)) - tfn) * log2(exp(1)) + 0.5 * log2(2.0 * pi * tfn) );
	return score;  // Change 0 to the final score you calculated
}

void pl2_tune(const std::shared_ptr<index::dblru_inverted_index>& idx, std::vector<corpus::document>& allqueries, index::ir_eval& eval)
{
	std::vector <double> cvalues = {0.3, 0.6, 0.9, 0.2, 0.1, 0.01, 2.1, 2.4};
	std::vector <double> lambdavalues = {0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,  1, 10};
	double maxmap = 0;
	double cmax = 7;
	double lambdamax = 0.1;
	std::ofstream outfile;
	auto ranker = make_unique<pl2_ranker>();

	for (auto& c : cvalues)  // Loops over all c values
	{
		for (auto& lambda : lambdavalues)  // Loops over all lambda values
		{
			ranker->set_param(c, lambda);
			for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query)
			{
				auto ranking = ranker->score(*idx, *query, 10);
				eval.avg_p(ranking, (*query).id(), 5);
			}
			outfile.open("pl2_results.txt", std::ios_base::app);
			outfile << c << "\t" << lambda << "\t" << eval.map() << "\n";
			outfile.close();
			if (eval.map() > maxmap)
			{
				maxmap = eval.map();
				cmax = c;
				lambdamax = lambda;
			}
			eval.reset_stats();
		}
	}
	std::cout << "Max MAP = " << maxmap << " achieved by " << "c = " << cmax << ", lambda = " << lambdamax << std::endl;
}

//MDTF2LN FUNCTION

class mdtf2ln_ranker: public index::ranker
{
private: // s_, mu_, alpha_ and lambda_ are the parameters of mdtf2ln
	double s_ = 0.2;
	double mu_ = 2000;
	double alpha_ = 0.3;
	double lambda_ = 0.7;

public:
	const static std::string id;
	mdtf2ln_ranker(); // Default Constructor
	mdtf2ln_ranker(double s, double mu, double alpha, double lambda); // Constructor which can set parameter values
	void set_param(double s, double mu, double alpha, double lambda){s_ = s; mu_ = mu; alpha_ = alpha; lambda_ = lambda;}; // Setter
	float score_one(const index::score_data&); // Calculates the score for a single matched term
	void save(std::ostream& out) const override;
};

const std::string mdtf2ln_ranker::id = "mdtf2ln"; // Used to identify mdtf2ln_ranker in config.toml
mdtf2ln_ranker::mdtf2ln_ranker(){}
mdtf2ln_ranker::mdtf2ln_ranker(double s, double mu, double alpha, double lambda) : s_{s}, mu_{mu}, alpha_{alpha}, lambda_{lambda} {}

void mdtf2ln_ranker::save(std::ostream& out) const
{
	io::packed::write(out, id);
	io::packed::write(out, s_);
	io::packed::write(out, mu_);
	io::packed::write(out, alpha_);
	io::packed::write(out, lambda_);
}

float mdtf2ln_ranker::score_one(const index::score_data& sd)
{
	double doc_len = sd.idx.doc_size(sd.d_id); // Current document length
	double avg_dl = sd.avg_dl; // Average document length in the corpus
	double tf = sd.doc_term_count; // Raw term count in the document
	double df = sd.doc_count; // number of docs that term appears in
	double pc = double(sd.corpus_term_count) / sd.total_terms;

	double s = s_; // mdtf2ln's parameter
	double mu = mu_; // mdtf2ln's parameter
	double alpha = alpha_; // mdtf2ln's parameter
	double lambda = lambda_; // mdtf2ln's parameter

	double tfok = 2.2 * tf / (1.2 + tf); // okapi tf term
	double idfpiv = std::log( (sd.num_docs + 1.0) / df );
	double tfidfdir = std::log( 1.0 + tf / ( mu * pc ) );
	double lnpiv = 1 - s + s * doc_len / avg_dl;

	double tfidf2 = alpha * tfok * idfpiv + (1.0 - alpha) * tfidfdir;

	double score = sd.query_term_weight * tfidf2 - sd.query_term_weight * std::pow(lnpiv,lambda);

	return score; // Change 0 to the final score you calculated
}



void mdtf2ln_tune (const std::shared_ptr<index::dblru_inverted_index> & idx, std::vector<corpus::document> & allqueries, index::ir_eval & eval)
{
	std::vector <double> alphavalues = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
	std::vector <double> lambdavalues = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8};
	std::vector <double> svalues = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
	std::vector <double> muvalues = {10.0, 50.0, 100.0};

	double maxmap = 0; // Stores the current maximum MAP value
	double smax = 0.2;
	double mumax = 2000;
	double alphamax = 0.3; // Stores the current optimal alpha (i.e. c that achieves max MAP) - Ignore the initial value
	double lambdamax = 0.7; // Stores the current optimal lambda - Ignore the initial value

	std::ofstream outfile;
	auto ranker = make_unique<mdtf2ln_ranker>(); // creates a pointer to a mdtf2ln_ranker instance

	for (auto& alpha : alphavalues) // Loops over all alpha values
	{
		for (auto& lambda : lambdavalues) // Loops over all lambda values
		{
			for (auto& s : svalues)
			{
				for (auto& mu : muvalues)
				{
					ranker->set_param(s, mu, alpha, lambda);
					for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query)
					{
						auto ranking = ranker->score(*idx, *query, 10);
						eval.avg_p(ranking,(*query).id(),5);
					}
					outfile.open("mdtf2ln_results.txt", std::ios_base::app);
					outfile << alpha << "\t" << lambda << "\t" << s << "\t" << mu << "\t" << eval.map() << "\n";
					outfile.close();
					if (eval.map() > maxmap)
					{
						maxmap = eval.map();
						alphamax = alpha;
						lambdamax = lambda;
						smax = s;
						mumax = mu;
					}
					eval.reset_stats();
				}
			}
		}
	}
	std::cout << "Max MAP = "<< maxmap << " achieved by " << "s = " << smax << ", mu = " << mumax << ", alpha = " << alphamax << ", lambda = " << lambdamax << std::endl;
}

class mptf2ln_ranker: public index::ranker
{
private: // s_, mu_, alpha_ and lambda_ are the parameters of mptf2ln
    double s_ = 0.2;
    double mu_ = 2000;
    double alpha_ = 0.3;
    double lambda_ = 0.7;

public:
    const static std::string id;
    mptf2ln_ranker(); // Default Constructor
    mptf2ln_ranker(double s, double mu, double alpha, double lambda); // Constructor which can set parameter values
    void set_param(double s, double mu, double alpha, double lambda){s_ = s; mu_ = mu; alpha_ = alpha; lambda_ = lambda;}; // Setter
    float score_one(const index::score_data&); // Calculates the score for a single matched term
	void save(std::ostream& out) const override;
};

const std::string mptf2ln_ranker::id = "mptf2ln"; // Used to identify mptf2ln_ranker in config.toml
mptf2ln_ranker::mptf2ln_ranker(){}
mptf2ln_ranker::mptf2ln_ranker(double s, double mu, double alpha, double lambda) : s_{s}, mu_{mu}, alpha_{alpha}, lambda_{lambda} {}

void mptf2ln_ranker::save(std::ostream& out) const
{
	io::packed::write(out, id);
	io::packed::write(out, s_);
	io::packed::write(out, mu_);
	io::packed::write(out, alpha_);
	io::packed::write(out, lambda_);
}

float mptf2ln_ranker::score_one(const index::score_data& sd)
{
    double doc_len = sd.idx.doc_size(sd.d_id); // Current document length
    double avg_dl = sd.avg_dl; // Average document length in the corpus
    double tf = sd.doc_term_count; // Raw term count in the document
    double df = sd.doc_count; // number of docs that term appears in
    double pc = double(sd.corpus_term_count) / sd.total_terms;

    double s = s_; // mptf2ln's parameter
    double mu = mu_; // mptf2ln's parameter
    double alpha = alpha_; // mptf2ln's parameter
    double lambda = lambda_; // mptf2ln's parameter

    double tfok = 2.2 * tf / ( 1.2 + tf ); // okapi tf term
    double idfpiv = std::log((sd.num_docs + 1.0) / df);
    double tfidfdir = std::log(1.0 + tf / (mu * pc));
    double lnpiv = 1 - s + s * doc_len / avg_dl;
    double tfidf2 = alpha * tfok * idfpiv + (1.0 - alpha) * tfidfdir;
    double score = sd.query_term_weight * tfidf2 / std::pow(lnpiv, lambda);

    return score; // Change 0 to the final score you calculated
}

void mptf2ln_tune (const std::shared_ptr<index::dblru_inverted_index> & idx, std::vector<corpus::document> & allqueries, index::ir_eval & eval)
{
	std::vector <double> alphavalues = {0.8, 1.0};
	std::vector <double> lambdavalues = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8};
	std::vector <double> svalues = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
	std::vector <double> muvalues = {10.0, 50.0, 100.0};

    double maxmap = 0; // Stores the current maximum MAP value
    double smax = 0.2;
    double mumax = 2000;
    double alphamax = 0.3; // Stores the current optimal alpha (i.e. c that achieves max MAP) - Ignore the initial value
    double lambdamax = 0.7;

    std::ofstream outfile;
    auto ranker = make_unique<mptf2ln_ranker>();

	for (auto& alpha : alphavalues) // Loops over all alpha values
	{
		for (auto& lambda : lambdavalues) // Loops over all lambda values
		{
			for (auto& s : svalues)
			{
				for (auto& mu : muvalues)
				{
					ranker->set_param(s, mu, alpha, lambda);
					for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query)
					{
						auto ranking = ranker->score(*idx, *query, 10);
						eval.avg_p(ranking,(*query).id(),5);
					}
					outfile.open("mptf2ln_results.txt", std::ios_base::app);
					outfile << alpha << "\t" << lambda << "\t" << s << "\t" << mu << "\t" << eval.map() << "\n";
					outfile.close();
					if (eval.map() > maxmap)
					{
						maxmap = eval.map();
						alphamax = alpha;
						lambdamax = lambda;
						smax = s;
						mumax = mu;
					}
					eval.reset_stats();
				}
			}
		}
	}
	std::cout << "Max MAP = "<< maxmap << " achieved by " << "s = " << smax << ", mu = " << mumax << ", alpha = " << alphamax << ", lambda = " << lambdamax << std::endl;
}

// YOUR METHOD TUNING

class sigmoidal_ranker: public index::ranker
{
private:
	double b1_ = 2.9;
	double b2_ = 3.7;
	double l1_ = 1.0;
	double l2_ = 1.0;
	double c_ = 0.5;
	double k_ = 1.0;

public:
    const static std::string id;
    sigmoidal_ranker(); // Default Constructor
    sigmoidal_ranker(double b1, double b2, double l1, double l2, double c, double k); // Constructor which can set parameter values
    void set_param(double b1, double b2, double l1, double l2, double c, double k){b1_ = b1; b2_ = b2; l1_ = l1; l2_ = l2; c_ = c; k_ = k;}; // Setter
    float score_one(const index::score_data&); // Calculates the score for a single matched term
	void save(std::ostream& out) const override;
};

const std::string sigmoidal_ranker::id = "sigmoidal_ranker"; // Used to identify mptf2ln_ranker in config.toml
sigmoidal_ranker::sigmoidal_ranker(){}
sigmoidal_ranker::sigmoidal_ranker(double b1, double b2, double l1, double l2, double c, double k) : b1_{b1}, b2_{b2}, l1_{l1}, l2_{l2}, c_{c}, k_{k} {}

void sigmoidal_ranker::save(std::ostream& out) const
{
	io::packed::write(out, id);
	io::packed::write(out, b1_);
	io::packed::write(out, b2_);
	io::packed::write(out, l1_);
	io::packed::write(out, l2_);
	io::packed::write(out, c_);
	io::packed::write(out, k_);
}

float sigmoidal_ranker::score_one(const index::score_data& sd)
{
    //PARAMETER DECLARATIONS

    double b1 = b1_;
	double b2 = b2_;
	double l1 = l1_;
	double l2 = l2_;
	double c = c_;
	double k = k_;

	//SCORE STRUCTURE

	double doc_len = sd.idx.doc_size(sd.d_id);
	double query_len = sd.query_length;
	double h = 1.0;

	if(doc_len < query_len)
	{
		double power = l1 * (doc_len - c * query_len);
		h = 1.0 + (b1 - 1.0)/(1.0 + std::exp(power));
	}
	if(doc_len > query_len)
	{
		double power = l2 * (doc_len - (1.0 + c) * query_len);
		h = 1.0 + (b2 - 1.0)/(1.0 + std::exp(-power));
	}

	double IDF = fastapprox::fastlog(1.0f + (sd.num_docs - sd.doc_count + 0.5f) / (sd.doc_count + 0.5f));
	float TF = ((k + 1.0f) * sd.doc_term_count) / ((k * h) + sd.doc_term_count);
    double score = sd.query_term_weight * TF * IDF;

    return score;
}

void sigmoidal_tune (const std::shared_ptr<index::dblru_inverted_index> & idx, std::vector<corpus::document> & allqueries, index::ir_eval & eval)
{
	std::vector <double> b1values = {1.0, 1.5, 2.0, 2.5};
	std::vector <double> b2values = {1.0, 1.5, 2.0, 2.5};
	std::vector <double> l1values = {0.25, 0.5, 1.0, 1.5, 1.75};
	std::vector <double> l2values = {0.25, 0.5, 1.0, 1.5, 1.75};
	std::vector <double> cvalues = {0.25, 0.5, 1.0, 1.5, 1.75};
	std::vector <double> kvalues = {0.01, 1.0, 2.0};

    std::ofstream outfile;
	std::ifstream infile("values.txt");
    auto ranker = make_unique<sigmoidal_ranker>();

	// for (auto& b1 : b1values) // Loops over all alpha values
	// {
	// 	for (auto& b2 : b2values) // Loops over all lambda values
	// 	{
	// 		for (auto& l1 : l1values)
	// 		{
	// 			for (auto& l2 : l2values)
	// 			{
	// 				for (auto& c : cvalues)
	// 				{
	// 					for (auto& k : kvalues)
	// 					{
	// 						ranker->set_param(b1, b2, l1, l2, c, k);
	// 						for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query)
	// 						{
	// 							auto ranking = ranker->score(*idx, *query, 10);
	// 							eval.avg_p(ranking,(*query).id(),5);
	// 						}
	// 						outfile.open("sigmoidal_results.txt", std::ios_base::app);
	// 						outfile << b1 << "\t" << b2 << "\t" << l1 << "\t" << l2 << "\t" << c << "\t" << k << "\t" << eval.map() << "\n";
	// 						outfile.close();
	// 						eval.reset_stats();
	// 					}
	// 				}
	// 			}
	// 		}
	// 	}
	// }

	double b1, b2, l1, l2, c, k;
	while (infile >> b1 >> b2 >> l1 >> l2 >> c >> k) {
		ranker->set_param(b1, b2, l1, l2, c, k);
		for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query)
		{
			auto ranking = ranker->score(*idx, *query, 10);
			eval.avg_p(ranking,(*query).id(),5);
		}
		outfile.open("sigmoidal_results.txt", std::ios_base::app);
		outfile << b1 << "\t" << b2 << "\t" << l1 << "\t" << l2 << "\t" << c << "\t" << k << "\t" << eval.map() << "\n";
		outfile.close();
		eval.reset_stats();
	}
}

class mountain_ranker: public index::ranker
{
private:
	double lambda_ = 1.0;
	double k_ = 1.0;

public:
    const static std::string id;
    mountain_ranker(); // Default Constructor
    mountain_ranker(double lambda, double k); // Constructor which can set parameter values
    void set_param(double lambda, double k){lambda_ = lambda; k_ = k;}; // Setter
    float score_one(const index::score_data&); // Calculates the score for a single matched term
	void save(std::ostream& out) const override;
};

const std::string mountain_ranker::id = "mountain_ranker"; // Used to identify mptf2ln_ranker in config.toml
mountain_ranker::mountain_ranker(){}
mountain_ranker::mountain_ranker(double lambda, double k) : lambda_{lambda}, k_{k} {}

void mountain_ranker::save(std::ostream& out) const
{
	io::packed::write(out, id);
	io::packed::write(out, lambda_);
	io::packed::write(out, k_);
}


float mountain_ranker::score_one(const index::score_data& sd)
{
    //PARAMETER DECLARATIONS

    double lambda = lambda_;
	double k = k_;

	double doc_len = sd.idx.doc_size(sd.d_id);
	double query_len = sd.query_length;
	double regulate = std::pow(std::abs(doc_len - query_len) + 1.0f, lambda);

	// double IDF = fastapprox::fastlog(1.0f + (sd.num_docs - sd.doc_count + 0.5f) / (sd.doc_count + 0.5f));
	// double TF = ((k + 1.0f) * sd.doc_term_count) / ((k * regulate) + sd.doc_term_count);
    // double score = sd.query_term_weight * TF * IDF;
	double TF = double(sd.doc_term_count) / regulate;
	double score = TF;
    return score;
}

void mountain_tune(const std::shared_ptr<index::dblru_inverted_index> & idx, std::vector<corpus::document> & allqueries, index::ir_eval & eval)
{
	std::vector <double> lambdavalues = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};
	std::vector <double> alphavalues = {0.8};
	std::vector <double> rvalues = {0.05};
	// std::vector <double> kvalues = {0.01, 1.0, 1.5, 2.0, 3.0, 6.0, 10.0, 25.0, 100.0, 200.0};
	std::ofstream outfile;
    auto ranker = make_unique<mountain_ranker>();

	for(auto& lambda : lambdavalues)
	{
		for(auto& alpha : alphavalues)
		{
			for(auto& r : rvalues)
			{
				ranker->set_param(lambda, 1.0);
				for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query)
				{
					auto ranking = ranker->score(*idx, *query, 20);
					//std::cout << "Before:\n";
					for (auto& result : ranking)
					{
						std::vector <std::string> correct_vector, incorrect_vector, intersect_vector, setdiff_correct, setdiff_incorrect;
						correct_vector = vec_dict[result.d_id];
						incorrect_vector = query_dict[(*query).id()];

						for (std::vector<std::string>::iterator i = correct_vector.begin(); i != correct_vector.end(); ++i)
						{
						    if (std::find(incorrect_vector.begin(), incorrect_vector.end(), *i) != incorrect_vector.end())
						        intersect_vector.push_back(*i);
						}
						//Set difference for correct_vector
						for (std::vector<std::string>::iterator i = correct_vector.begin(); i != correct_vector.end(); ++i)
						{
						    if (std::find(intersect_vector.begin(), intersect_vector.end(), *i) == intersect_vector.end())
						        setdiff_correct.push_back(*i);
						}
						//Set difference for incorrect_vector
						for (std::vector<std::string>::iterator i = incorrect_vector.begin(); i != incorrect_vector.end(); ++i)
						{
						    if (std::find(intersect_vector.begin(), intersect_vector.end(), *i) == intersect_vector.end())
						        setdiff_incorrect.push_back(*i);
						}

						std::vector <std::string> maps;
						std::vector <std::string> first, second;
						first = setdiff_incorrect;
						second = setdiff_correct;

						if(first.empty())
							first.push_back("nun");

						if(second.empty())
							second.push_back("nun");

						while(first.size() < second.size())
						{
							int pos = std::ceil(first.size() / 2.0);
							first.insert(first.begin() + pos, "nun");
						}

						while(first.size() > second.size())
						{
							int pos = std::floor(second.size() / 2.0);
							second.insert(second.begin() + pos, "nun");
						}

						for(unsigned int i = 0; i < first.size(); i++)
							maps.push_back(first[i] + "-" + second[i]);

						for(unsigned int i = 0; i < first.size() - 1; i++)
							maps.push_back(first[i] + "-" + second[i + 1]);

						for(unsigned int i = 1; i < first.size(); i++)
							maps.push_back(first[i] + "-" + second[i - 1]);

						double point_score = 0.0;
						for(auto& i : maps)
						{
							double curr_score =  static_cast<double>(pointer_dict[i]);
							point_score += std::pow(curr_score, r);
						}
						// std::cout << result.score << "\t" << alpha * result.score + (1.0 - alpha) * point_score << "\t" << idx->doc_path(result.d_id) << std::endl;
						//result.score = alpha * result.score + (1.0 - alpha) * point_score;
					}
					// std::sort(ranking.begin(), ranking.end(), [](const auto& a, const auto& b) {return a.score > b.score;});
					// for(auto& result : ranking)
					// {
					// 	std::cout << idx->doc_path(result.d_id) << std::endl;
					// }
					// std::cout << std::endl;
					eval.avg_p(ranking, (*query).id(), 20);
				}
				outfile.open("mountain_results_less_data.txt", std::ios_base::app);
				//outfile << lambda << "\t" << r << "\t" << alpha << "\t" << eval.map() << "\n";
				outfile << lambda << "\t"  << eval.map() << "\n";
				outfile.close();
				eval.reset_stats();
			}
		}
	}
}

namespace meta
{
	namespace index
	{
		template <>
		std::unique_ptr<ranker> make_ranker<pl2_ranker>(const cpptoml::table& config)
		{
			return make_unique<pl2_ranker>(config.get_as<double>("c").value_or(7.0f), config.get_as<double>("lambda").value_or(0.1f));
		}

		template <>
		std::unique_ptr<ranker> make_ranker<mountain_ranker>(const cpptoml::table& config)
		{
			return make_unique<mountain_ranker>(config.get_as<double>("lambda").value_or(1.0f), config.get_as<double>("k").value_or(1.0f));
		}

		template <>
		std::unique_ptr<ranker> make_ranker<sigmoidal_ranker>(const cpptoml::table& config)
		{
			return make_unique<sigmoidal_ranker>(config.get_as<double>("b1").value_or(1.0f), config.get_as<double>("b2").value_or(1.0f),
			config.get_as<double>("l1").value_or(1.0f), config.get_as<double>("l2").value_or(1.0f),
			config.get_as<double>("c").value_or(1.0f), config.get_as<double>("k").value_or(1.0f));
		}

		template <>
		std::unique_ptr<ranker> make_ranker<mdtf2ln_ranker>(const cpptoml::table & config)
		{
			double s = 0.2;
			if (auto s_file = config.get_as<double>("s"))
				s = *s_file;

			double mu = 2000;
			if (auto mu_file = config.get_as<double>("mu"))
				mu = *mu_file;

			double alpha = 0.3;
			if (auto alpha_file = config.get_as<double>("alpha"))
				alpha = *alpha_file;

			double lambda = 0.7;
			if (auto lambda_file = config.get_as<double>("lambda"))
				lambda = *lambda_file;

			return make_unique<mdtf2ln_ranker>(s, mu, alpha, lambda);
		}

		template <>
		std::unique_ptr<ranker>make_ranker<mptf2ln_ranker>(const cpptoml::table & config)
		{
		    double s = 0.2;
		    if (auto s_file = config.get_as<double>("s"))
		        s = *s_file;

		    double mu = 2000;
		    if (auto mu_file = config.get_as<double>("mu"))
		        mu = *mu_file;

		    double alpha = 0.3;
		    if (auto alpha_file = config.get_as<double>("alpha"))
		        alpha = *alpha_file;

		    double lambda = 0.7;
		    if (auto lambda_file = config.get_as<double>("lambda"))
		        lambda = *lambda_file;

		    return make_unique<mptf2ln_ranker>(s, mu, alpha, lambda);
		}
	}
}

// BM25 TUNING

void bm25_tune(const std::shared_ptr<index::dblru_inverted_index> & idx, std::vector<corpus::document> & allqueries, index::ir_eval & eval)
{
	std::vector <double> k1values = {0.01, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3};
	std::vector <double> bvalues = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	std::vector <double> k3values = {500.0};
	std::ofstream outfile;
	double maxmap = 0; // Stores the current maximum MAP value
	double k1max = 1.2;
	double bmax = 0.75;
	double k3max = 500.0;

	for (auto & k1 : k1values) // Loops over all k1 values
	{
		for (auto & b : bvalues) // Loops over all b values
		{
			for (auto & k3 : k3values) // Loops over all k3 values
			{
				auto ranker = make_unique<meta::index::okapi_bm25>(k1, b, k3);
				for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query)
				{
					auto ranking = ranker->score(*idx, *query, 10);
					eval.avg_p(ranking,(*query).id(),5);
				}
				outfile.open("bm25_results.txt", std::ios_base::app);
				outfile << k1 << "\t" << b << "\t" << k3 << "\t" << eval.map() << "\n";
				outfile.close();
				if(eval.map() > maxmap)
				{
					maxmap = eval.map();
					k1max = k1;
					k3max = k3;
					bmax = b;
				}
				eval.reset_stats();
			}
		}
	}
	std::cout << "The answer is k1 : " << k1max << ", b : " << bmax << ", k3 : " << k3max << ", with the MAP of " << maxmap << std::endl;
}

void pl_tune(const std::shared_ptr<index::dblru_inverted_index> & idx, std::vector<corpus::document> & allqueries, index::ir_eval & eval)
{
	std::vector <double> svalues = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	std::ofstream outfile;
	double maxmap = 0;
	double smax = 0.0;

	for (auto & s : svalues)
	{
		auto ranker = make_unique<meta::index::pivoted_length>(s);
		for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query)
		{
			auto ranking = ranker->score(*idx, *query, 10);
			eval.avg_p(ranking,(*query).id(),5);
		}
		outfile.open("pl_plainbi_results.txt", std::ios_base::app);
		outfile << s << "\t" << eval.map() << "\n";
		outfile.close();
		if(eval.map() > maxmap)
		{
			maxmap = eval.map();
			smax = s;
		}
		eval.reset_stats();
	}
	std::cout << "The answer is s : " << smax << ", with the MAP of " << maxmap << std::endl;
}

void jm_tune(const std::shared_ptr<index::dblru_inverted_index> & idx, std::vector<corpus::document> & allqueries, index::ir_eval & eval)
{
	std::vector <double> lambdavalues = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	std::ofstream outfile;
	double maxmap = 0;
	double lambdamax = 0.0;

	for (auto & lambda : lambdavalues)
	{
		auto ranker = make_unique<meta::index::jelinek_mercer>(lambda);
		for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query)
		{
			auto ranking = ranker->score(*idx, *query, 10);
			eval.avg_p(ranking, (*query).id(), 5);
		}
		outfile.open("jm_results.txt", std::ios_base::app);
		outfile << lambda << "\t" << eval.map() << "\n";
		outfile.close();
		if(eval.map() > maxmap)
		{
			maxmap = eval.map();
			lambdamax = lambda;
		}
		eval.reset_stats();
	}
	std::cout << "The answer is lambda : " << lambdamax << ", with the MAP of " << maxmap << std::endl;
}

void dirichlet_tune(const std::shared_ptr<index::dblru_inverted_index> & idx, std::vector<corpus::document> & allqueries, index::ir_eval & eval)
{
	std::vector <double> muvalues = {0.0, 500.0, 2000.0};
	std::ofstream outfile;
	double maxmap = 0;
	double mumax = 0.0;

	for (auto & mu : muvalues)
	{
		auto ranker = make_unique<meta::index::dirichlet_prior>(mu);
		double mean_ndcg = 0.0;
		for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query)
		{
			auto ranking = ranker->score(*idx, *query, 10);
			// std::cout << "Results for query " << (*query).id() << std::endl;
			// auto result_num = 1;
			// for (auto& result : ranking)
			// {
			// 	print_results(idx, result, result_num);
			// 	result_num++;
			// }
			eval.avg_p(ranking,(*query).id(),10);
			mean_ndcg += eval.ndcg(ranking,(*query).id(),10);
			// eval.print_stats(ranking, (*query).id());
		}
		outfile.open("dirichlet_05data_gene_results.txt", std::ios_base::app);
		outfile << mu << "\t" << eval.map() << "\t" << mean_ndcg / 1000.0 << "\n";
		outfile.close();
		if(eval.map() > maxmap)
		{
			maxmap = eval.map();
			mumax = mu;
		}
		eval.reset_stats();
	}
	std::cout << "The answer is mu : " << mumax << ", with the MAP of " << maxmap << std::endl;
}

// void discount_tune(const std::shared_ptr<index::dblru_inverted_index> & idx, std::vector<corpus::document> & allqueries, index::ir_eval & eval)
// {
// 	std::vector <double> deltavalues = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
// 	std::ofstream outfile;
// 	double maxmap = 0;
// 	double deltamax = 0.0;
//
// 	for (auto & delta : deltavalues)
// 	{
// 		auto ranker = make_unique<meta::index::ranker::absolute_discount>(delta);
// 		for (std::vector<corpus::document>::iterator query = allqueries.begin(); query != allqueries.end(); ++query)
// 		{
// 			auto ranking = ranker->score(*idx, *query, 10);
// 			eval.avg_p(ranking,(*query).id(),5);
// 		}
// 		outfile.open("discount_results.txt", std::ios_base::app);
// 		outfile << delta << "\t" << eval.map() << "\n";
// 		outfile.close();
// 		if(eval.map() > maxmap)
// 		{
// 			maxmap = eval.map();
// 			deltamax = delta;
// 		}
// 		eval.reset_stats();
// 	}
// 	std::cout << "The answer is delta : " << deltamax << ", with the MAP of " << maxmap << std::endl;
// }

int main(int argc, char* argv[])
{

	if (argc != 2)
	{
		std::cerr << "Usage:\t" << argv[0] << " config.toml" << std::endl;
		return 1;
	}

	// Log to standard error
	logging::set_cerr_logging();

	// Register additional analyzers
	parser::register_analyzers();
	sequence::register_analyzers();

	// std::ifstream list, incorrect,point;
	// list.open("bigrams-no-offset.txt");
	// incorrect.open("incorrect.txt");
	// point.open("pointer-list.txt");
	// std::string current;
	// //Read from file and Split
	// while(std::getline(list, current))
	// {
	// 	std::vector <std::string> temp;
	// 	std::istringstream iss(current);
	// 	std::string tok;
	// 	while(std::getline(iss, tok, ' '))
	// 		temp.push_back(tok);
	// 	vec_dict.push_back(temp);
	// }
	// while(std::getline(incorrect, current))
	// {
	// 	std::vector <std::string> temp;
	// 	std::istringstream iss(current);
	// 	std::string tok;
	// 	while(std::getline(iss, tok, ' '))
	// 		temp.push_back(tok);
	// 	query_dict.push_back(temp);
	// }
	// while(std::getline(point, current))
	// {
	// 	std::istringstream iss_c(current);
	// 	std::string strtok, inttok;
	// 	std::getline(iss_c, strtok, ' ');
	// 	std::getline(iss_c, inttok, ' ');
	// 	pointer_dict[strtok] = std::stoi(inttok); //Insert into dictionary
	// }

	// Create an inverted index based on the config file
	auto config = cpptoml::parse_file(argv[1]);
	auto idx = index::make_index<index::dblru_inverted_index>(*config, 30000);
	// Get the path to the file containing queries
	auto query_path = config->get_as<std::string>("query-path");
	if (!query_path)
		throw std::runtime_error{
			"config file needs a \"query-path\" parameter"};
	std::ifstream queries{*query_path};
	auto eval = index::ir_eval(*config);
	std::string content;
	std::vector<corpus::document> allqueries;
	size_t i = 0;
	while (queries.good() && i < 1000) {
		std::getline(queries, content);
		corpus::document query{doc_id{i}};
		query.content(content);
		allqueries.push_back(query);
		i++;
	}
	dirichlet_tune(idx, allqueries, eval);
}

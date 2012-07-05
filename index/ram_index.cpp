/**
 * @file ram_index.cpp
 */

#include "ram_index.h"

RAMIndex::RAMIndex(const vector<string> & indexFiles, Tokenizer* tokenizer)
{
    cout << "[RAMIndex]: creating index from " << indexFiles.size() << " files" << endl;

    _docFreqs = unordered_map<TermID, unsigned int>();
    _documents = vector<Document>();
    _avgDocLength = 0;
    
    size_t docNum = 0;
    for(auto & file: indexFiles)
    {
        Document document(file);
        tokenizer->tokenize(document, &_docFreqs);
        _documents.push_back(document);
        _avgDocLength += document.getLength();

        if(docNum++ % 10 == 0)
            cout << "  " << ((double) docNum / indexFiles.size() * 100) << "%    \r";
    }
    cout << "  100%        " << endl;

    _avgDocLength /= _documents.size();
}

RAMIndex::RAMIndex(const vector<Document> & indexDocs, Tokenizer* tokenizer)
{
    cout << "[RAMIndex]: creating index from " << indexDocs.size() << " Documents" << endl;

    _docFreqs = unordered_map<TermID, unsigned int>();
    _documents = indexDocs;
    _avgDocLength = 0;
    size_t docNum = 0;
    for(auto & doc: _documents)
    {
        //combineMap(doc.getFrequencies()); // call this if doc was already tokenized
        tokenizer->tokenize(doc, &_docFreqs);
        _avgDocLength += doc.getLength();
        if(docNum++ % 10 == 0)
            cout << "  " << ((double) docNum / _documents.size() * 100) << "%    \r";
    }
    cout << "  100%        " << endl;

    _avgDocLength /= _documents.size();
}

void RAMIndex::combineMap(const unordered_map<TermID, unsigned int> & newFreqs)
{
    for(auto & freq: _docFreqs)
        _docFreqs[freq.first] += freq.second;
}

double RAMIndex::scoreDocument(const Document & document, const Document & query) const
{
    double score = 0.0;
    double k1 = 1.5;
    double b = 0.75;
    double k3 = 500;
    double docLength = document.getLength();
    double numDocs = _documents.size();

    const unordered_map<TermID, unsigned int> frequencies = query.getFrequencies();
    for(auto & term: frequencies)
    {
        auto df = _docFreqs.find(term.first);
        double docFreq = (df == _docFreqs.end()) ? (0.0) : (df->second);
        double termFreq = document.getFrequency(term.first);
        double queryTermFreq = query.getFrequency(term.first);

        double IDF = log((numDocs - docFreq + 0.5) / (docFreq + 0.5));
        double TF = ((k1 + 1.0) * termFreq) / ((k1 * ((1.0 - b) + b * docLength / _avgDocLength)) + termFreq);
        double QTF = ((k3 + 1.0) * queryTermFreq) / (k3 + queryTermFreq);

        cerr << term.first << ": IDF: " << IDF << ", TF: " << TF << ", QTF: " << QTF << endl;
        score += IDF * TF * QTF;
    }

    return score;
}

size_t RAMIndex::getAvgDocLength() const
{
    return _avgDocLength;
}

multimap<double, string> RAMIndex::search(const Document & query) const
{
    cout << "[RAMIndex]: scoring documents for query " << query.getName()
         << " (" << query.getCategory() << ")" << endl;

    multimap<double, string> ranks;
    #pragma omp parallel for
    for(size_t idx = 0; idx < _documents.size(); ++idx)
    {
        double score = scoreDocument(_documents[idx], query);
        if(score != 0.0)
        {
            #pragma omp critical
            {
                ranks.insert(make_pair(score, _documents[idx].getName() + " (" + _documents[idx].getCategory() + ")"));
            }
        }
    }

    return ranks;
}

string RAMIndex::classifyKNN(const Document & query, size_t k) const
{
    multimap<double, string> ranking = search(query);
    unordered_map<string, size_t> counts;
    size_t numResults = 0;
    for(auto result = ranking.rbegin(); result != ranking.rend() && numResults++ != k; ++result)
    {
        size_t space = result->second.find_first_of(" ") + 1;
        string category = result->second.substr(space, result->second.size() - space);
        counts[category]++;
    }

    string best = "[no results]";
    size_t high = 0;
    for(auto & count: counts)
    {
        if(count.second > high)
        {
            best = count.first;
            high = count.second;
        }
    }

    return best;
}
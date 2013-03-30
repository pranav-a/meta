/**
 * @file knn.h
 * Collection of k-nearest neighbor tools.
 */

#ifndef _KNN_H_
#define _KNN_H_

#include <map>
#include <unordered_map>
#include <memory>
#include <vector>
#include <string>
#include "index/index.h"

namespace meta {
namespace classify {

/**
 * Runs KNN on a single index or multiple indexes.
 */
namespace knn
{
    /**
     * Runs a KNN classifier.
     * @param query - the query to run
     * @param index - the index to perform the KNN on
     * @param k - the value of k in KNN
     */
    std::string classify(index::Document & query, std::shared_ptr<index::Index> index, size_t k);

    /**
     * Runs a KNN classifier on multiple indexes.
     * @param query - the query to run
     * @param indexes - the indexes to perform the KNN search on
     * @param weights - ensemble linear interpolation weights
     * @param k - the value of k in kNN
     */
    std::string classify(index::Document & query,
            std::vector<std::shared_ptr<index::Index>> indexes,
            std::vector<double> weights, size_t k);

    /**
     * Helper functions for the knn namespace.
     */
    namespace internal
    {
        /**
         * Normalizes the values in scores to be in [0, 1].
         * @param scores - the scores to normalize
         * @return the normalized scores
         */
        std::unordered_map<std::string, double>
        normalize(const std::multimap<double, std::string> & scores);

        /**
         * Finds the most common occurrence in the top k results.
         * @param rankings - ranked list of documents returned by a search
         *  engine
         * @param k - k value in kNN
         * @return the class label for the most common document
         */
        std::string findNN(const std::multimap<double, std::string> & rankings, size_t k);

        /**
         * Used for tiebreaking. If there are the same number of a certain class, prefer the class
         *  that was seen first.
         * @param check
         * @param best
         * @param orderSeen
         * @return if the class to check should be ranked about the current best
         */
        bool isHigherRank(const std::string & check,
                const std::string & best,
                const std::vector<std::string> & orderSeen);
    }

    /**
     * Basic exception for KNN interactions.
     */
    class knn_exception: public std::exception
    {
        public:
            
            knn_exception(const std::string & error):
                _error(error) { /* nothing */ }

            const char* what () const throw ()
            {
                return _error.c_str();
            }
       
        private:
       
            std::string _error;
    };
}

}
}

#endif

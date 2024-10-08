#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <omp.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "efanna2e/distance.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "index_bipartite.h"

namespace py = pybind11;

class RoarGraph {
private:
    uint32_t M_sq;
    uint32_t L_pq;
    uint32_t M_pjbp, L_pjpq;
    uint32_t K;
    uint32_t num_threads;
    efanna2e::IndexBipartite* index_bipartite = nullptr;
    efanna2e::Parameters* parameters;
    uint32_t n_vectors;
    uint32_t n_queries;
    uint32_t dim;
    float* v_ptr;
    float* sq_ptr;

    double knn_time = 0;
    double graph_time = 0;
public:
    RoarGraph(uint32_t M_sq, uint32_t L_pq, uint32_t M_pjbp, uint32_t L_pjpq, uint32_t K, uint32_t num_threads) : 
        M_sq(M_sq), L_pq(L_pq), M_pjbp(M_pjbp), L_pjpq(L_pjpq), K(K), num_threads(num_threads) {};

    ~RoarGraph() {
        delete index_bipartite;
        delete parameters;
        delete[] v_ptr;
        delete[] sq_ptr;
    }
    
    void build(py::array_t<float>& vectors, py::array_t<float>& sample_queries) {
        py::buffer_info v_buf = vectors.request();
        py::buffer_info sq_buf = sample_queries.request();
        if (v_buf.ndim != 2) {
            throw std::runtime_error("numpy.ndarray dims must be 2!");
        }
        this->n_vectors = v_buf.shape[0];
        this->n_queries = sq_buf.shape[0];
        this->dim = v_buf.shape[1];
        // copy data to class member
        this->v_ptr = new float[this->n_vectors * this->dim];
        this->sq_ptr = new float[this->n_queries * this->dim];
        std::memcpy(this->v_ptr, v_buf.ptr, this->n_vectors * this->dim * sizeof(float));
        std::memcpy(this->sq_ptr, sq_buf.ptr, this->n_queries * this->dim * sizeof(float));

        // efanna2e::IndexBipartite index_bipartite(base_dim, base_num + sq_num, dist_metric, nullptr);
        this->index_bipartite = new efanna2e::IndexBipartite(this->dim, this->n_vectors + this->n_queries, efanna2e::INNER_PRODUCT, nullptr);

        this->parameters = new efanna2e::Parameters();
        parameters->Set<uint32_t>("M_sq", this->M_sq);
        parameters->Set<uint32_t>("L_pq", this->L_pq);
        parameters->Set<uint32_t>("M_pjbp", this->M_pjbp);
        parameters->Set<uint32_t>("L_pjpq", this->L_pjpq);
        parameters->Set<uint32_t>("num_threads", this->num_threads);
        // std::cout << "M_bp: " << M_bp << std::endl;
        // index_bipartite.LoadLearnBaseKNN(learn_base_nn_file.c_str());

        auto s = std::chrono::high_resolution_clock::now();
        index_bipartite->ComputeLearnBaseKNN(this->v_ptr, this->sq_ptr, this->K, efanna2e::INNER_PRODUCT, this->n_vectors, this->n_queries, this->dim);
        auto e = std::chrono::high_resolution_clock::now();
        e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        this->knn_time = diff.count();
        // std::cout << "BaseKNN construction time: " << this->knn_time << "\n";

        omp_set_num_threads(this->num_threads);
        s = std::chrono::high_resolution_clock::now();
        index_bipartite->BuildRoarGraph(this->n_queries, this->sq_ptr, this->n_vectors, this->v_ptr, *this->parameters);
        e = std::chrono::high_resolution_clock::now();
        diff = e - s;
        this->graph_time = diff.count();
        // std::cout << "RoarGraph indexing time: " << this->graph_time << "\n";
    }

    py::array_t<uint32_t> search(py::array_t<float>& query, int k) {
        py::buffer_info q_buf = query.request();
        float* q_ptr = static_cast<float*>(q_buf.ptr);
        if (q_buf.ndim != 1) {
            throw std::runtime_error("numpy.ndarray dims must be 1!");
        }

        // std::cout << "Searching for " << k << " nearest neighbors" << std::endl;

        index_bipartite->InitVisitedListPool(this->num_threads);

        unsigned res[k];
        std::vector<float> dists(k);        // useless
        size_t unused = 0;      // useless index
        index_bipartite->SearchRoarGraph(q_ptr, k, unused, *(this->parameters), res, dists);

        // std::cout << "Search done" << std::endl;

        py::array_t<uint32_t> result = py::array_t<uint32_t>(k);
        py::buffer_info result_buf = result.request();
        uint32_t* result_ptr = static_cast<uint32_t*>(result_buf.ptr);

        for (int i = 0; i < k; i++) {
            result_ptr[i] = res[i];
        }

        return result;
    }

    py::array_t<uint32_t> searchIPDiff(py::array_t<float>& query, float exp_ratio) {
        py::buffer_info q_buf = query.request();
        float* q_ptr = static_cast<float*>(q_buf.ptr);
        if (q_buf.ndim != 1) {
            throw std::runtime_error("numpy.ndarray dims must be 1!");
        }
        if (exp_ratio <= 0 || exp_ratio > 1) {
            throw std::runtime_error("exp_ratio must be in (0, 1]");
        }

        // std::cout << "Searching vectors that larger that exp_max * " << exp_ratio << std::endl;

        index_bipartite->InitVisitedListPool(this->num_threads);

        std::vector<unsigned> res;
        std::vector<float> dists;        // useless
        size_t unused = 0;      // useless index
        index_bipartite->SearchRoarGraphIPDiff(q_ptr, exp_ratio, unused, *(this->parameters), res, dists);

        // std::cout << "Search done" << std::endl;

        py::array_t<uint32_t> result = py::array_t<uint32_t>(res.size());
        py::buffer_info result_buf = result.request();
        uint32_t* result_ptr = static_cast<uint32_t*>(result_buf.ptr);

        for (int i = 0; i < res.size(); i++) {
            result_ptr[i] = res[i];
        }

        return result;
    }

    double getKnnTime() {
        return this->knn_time;
    }

    double getGraphTime() {
        return this->graph_time;
    }
};

PYBIND11_MODULE( roargraph, m ){
    m.doc() = "RoarGraph";

    pybind11::class_<RoarGraph>(m, "RoarGraph" )
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(),
            py::arg("M_sq"), py::arg("L_pq"), py::arg("M_pjbp"), py::arg("L_pjpq"), py::arg("K"), py::arg("num_threads"))
        .def("build", &RoarGraph::build, 
            py::arg("vectors"), py::arg("sample_queries"))
        .def("search", &RoarGraph::search, 
            py::arg("query"), py::arg("k"))
        .def("searchIPDiff", &RoarGraph::searchIPDiff, 
            py::arg("query"), py::arg("exp_ratio"))
        .def("getKnnTime", &RoarGraph::getKnnTime, 
            "Get the time for building base KNN")
        .def("getGraphTime", &RoarGraph::getGraphTime, 
            "Get the time for building RoarGraph");
}
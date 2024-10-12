
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


class RoarGraph {
public:
    uint32_t M_sq;
    uint32_t L_pq;
    uint32_t M_pjbp, L_pjpq;
    uint32_t K;
    uint32_t num_threads;
    efanna2e::IndexBipartite* index_bipartite = nullptr;
    efanna2e::Parameters parameters;
    uint32_t n_vectors;
    uint32_t n_queries;
    uint32_t dim;
    float* v_ptr;
    float* sq_ptr;

    double knn_time = 0;
    double graph_time = 0;


    RoarGraph(uint32_t M_sq, uint32_t L_pq, uint32_t M_pjbp, uint32_t L_pjpq, uint32_t K, uint32_t num_threads) :
        M_sq(M_sq), L_pq(L_pq), M_pjbp(M_pjbp), L_pjpq(L_pjpq), K(K), num_threads(num_threads) {};

    ~RoarGraph() {
        delete index_bipartite;
        // delete parameters;
    }

    void build(std::vector<float>& vectors, std::vector<float>& sample_queries, int dim) {
        // py::buffer_info v_buf = vectors.request();
        // py::buffer_info sq_buf = sample_queries.request();
        // if (v_buf.ndim != 2) {
        //     throw std::runtime_error("numpy.ndarray dims must be 2!");
        // }
        this->n_vectors = vectors.size()/dim;
        this->n_queries = sample_queries.size()/dim;
        this->dim = dim;
        // copy data to class member
        this->v_ptr = new float[this->n_vectors * this->dim];
        this->sq_ptr = new float[this->n_queries * this->dim];
        std::memcpy(this->v_ptr, vectors.data(), this->n_vectors * this->dim * sizeof(float));
        std::memcpy(this->sq_ptr, sample_queries.data(), this->n_queries * this->dim * sizeof(float));
        // efanna2e::IndexBipartite index_bipartite(base_dim, base_num + sq_num, dist_metric, nullptr);
        this->index_bipartite = new efanna2e::IndexBipartite(this->dim, this->n_vectors + this->n_queries, efanna2e::INNER_PRODUCT, nullptr);

        std::cout << "num_threads: " << this->num_threads << std::endl;
        parameters.Set<uint32_t>("M_sq", this->M_sq);
        parameters.Set<uint32_t>("L_pq", this->L_pq);
        parameters.Set<uint32_t>("M_pjbp", this->M_pjbp);
        parameters.Set<uint32_t>("L_pjpq", this->L_pjpq);
        parameters.Set<uint32_t>("num_threads", this->num_threads);
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
        index_bipartite->BuildRoarGraph(this->n_queries, this->sq_ptr, this->n_vectors, this->v_ptr, parameters);
        e = std::chrono::high_resolution_clock::now();
        diff = e - s;
        this->graph_time = diff.count();
        // std::cout << "RoarGraph indexing time: " << this->graph_time << "\n";
    }

    std::vector<uint32_t> search(const float *query, int k) {
        // py::buffer_info q_buf = query.request();
        // float* q_ptr = static_cast<float*>(q_buf.ptr);
        // if (q_buf.ndim != 1) {
        //     throw std::runtime_error("numpy.ndarray dims must be 1!");
        // }

        // std::cout << "Searching for " << k << " nearest neighbors" << std::endl;
        index_bipartite->InitVisitedListPool(this->num_threads);

        unsigned res[k];
        std::vector<float> dists(k);        // useless
        size_t unused = 0;      // useless index
        index_bipartite->SearchRoarGraph(query, k, unused, parameters, res, dists);

        // std::cout << "Search done" << std::endl;

        std::vector<uint32_t> result(k);

        for (int i = 0; i < k; i++) {
            result[i] = res[i];
        }

        return result;
    }

    std::vector<uint32_t> search_by_threshold(std::vector<float>& query, float exp_ratio, int update_cnt_threshold, int capacity_factor) {
        // py::buffer_info q_buf = query.request();
        // float* q_ptr = static_cast<float*>(q_buf.ptr);
        // if (q_buf.ndim != 1) {
        //     throw std::runtime_error("numpy.ndarray dims must be 1!");
        // }
        // if (exp_ratio <= 0 || exp_ratio > 1) {
        //     throw std::runtime_error("exp_ratio must be in (0, 1]");
        // }

        // std::cout << "Searching vectors that larger that exp_max * " << exp_ratio << std::endl;

        index_bipartite->InitVisitedListPool(this->num_threads);

        std::vector<unsigned> res;
        std::vector<float> dists;        // useless
        size_t unused = 0;      // useless index
        index_bipartite->SearchRoarGraphThreshold(query.data(), exp_ratio, update_cnt_threshold, capacity_factor, unused, parameters, res, dists);

        // std::cout << "Search done" << std::endl;

        std::vector<uint32_t> result={};

        for (int i = 0; i < res.size(); i++) {
            result.emplace_back(res[i]);
        }

        return result;
    }

    double get_knn_time() {
        return this->knn_time;
    }

    double get_graph_time() {
        return this->graph_time;
    }
};

// layer, head, token * dim
typedef std::vector<std::vector<std::vector<float>>> VectorMatrix;

void Load(std::string path, 
          VectorMatrix &total_queries, 
          VectorMatrix &total_keys,
          VectorMatrix &total_sample_queries,
          int sample_size,
          int &dim, int &n_layer, int &n_head, int &n_token, int &decode_pos){

    std::string info_file = path + "/info.txt";
    std::ifstream info_f(info_file);
    std::string line;
    int cnt = 0; 
    while (std::getline(info_f, line)) {
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        if (key == "dim:") {
            iss >> dim;
        } else if (key == "n_layer:") {
            iss >> n_layer;
        } else if (key == "n_head:") {
            iss >> n_head;
        } else if (key == "n_token:") {
            iss >> n_token;
        } else if (key == "decode_pos:") {
            iss >> decode_pos;
        } else {
            throw std::runtime_error("Unknown key: " + key);
        }
        cnt++;
    }
    if (cnt != 5) {
        throw std::runtime_error("info.txt should have 5 lines");
    }

    // read binary fvecs files
    // each file has n_layer * n_head * n_token vectors, each vector has dim floats
    // we use first sample_size vectors as sample queries
    std::string queries_filepath = path + "/queries.fvecs";
    std::string keys_filepath = path + "/keys.fvecs";
    std::ifstream queries_f(queries_filepath, std::ios::binary);
    std::ifstream keys_f(keys_filepath, std::ios::binary);
    if (sample_size > n_token) {
        throw std::runtime_error("sample_size should be less than n_token");
    }

    for (int i = 0; i < n_layer; i++) {
        std::vector<std::vector<float>> layer_queries, layer_keys, layer_sample_queries;
        for (int j = 0; j < n_head; j++) {
            std::vector<float> queries, keys, sample_queries;      // for one head
            // each head has n_token * dim floats
            queries.resize(n_token * dim);
            keys.resize(n_token * dim);
            queries_f.read((char*)queries.data(), n_token * dim * sizeof(float));
            keys_f.read((char*)keys.data(), n_token * dim * sizeof(float));
            layer_queries.push_back(queries);
            layer_keys.push_back(keys);

            // peek first sample_size vectors
            sample_queries.resize(sample_size * dim);
            std::copy(queries.begin(), queries.begin() + sample_size * dim, sample_queries.begin());
            layer_sample_queries.push_back(sample_queries);
        }
        total_queries.push_back(layer_queries);
        total_keys.push_back(layer_keys);
        total_sample_queries.push_back(layer_sample_queries);
    }
}

inline bool to_test(int layer, int head) {
    // return layer >= 8 && layer < 9 && head < 1;
    // return layer == 4 and head == 1;
    return true;
}

int main(){
    std::cout << "START!!!" << std::endl;

    int topK = 64, sample_size = 100;
    std::string path = "../../../../../offloaded_data/longbench_qasper_q0";

    VectorMatrix queries, keys, sample_queries;
    int dim, n_layer, n_head, n_token, decode_pos;
    Load(path, queries, keys, sample_queries, sample_size, dim, n_layer, n_head, n_token, decode_pos);
    // we have check sample_size > n_token in Load
    
    // graph matrix n_layer * n_head
    std::vector<RoarGraph*> graphs(n_layer * n_head);
    for (int i = 0; i < n_layer; i++) {
        for (int j = 0; j < n_head; j++) {
            if (!to_test(i, j)) {
                continue;
            }
            std::cout << "Building graph for layer " << i << " head " << j << std::endl;
            auto s = std::chrono::high_resolution_clock::now();
            RoarGraph *graph = new RoarGraph(100, 100, 35, 500, 100, 64);
            graph->build(keys[i][j], sample_queries[i][j], dim);
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            std::cout << "Graph construction time: " << diff.count() << "\n\n";
            graphs[i * n_head + j] = graph;
        }
    }
    
    // search
    for (int i = 0; i < n_layer; i++) {
        for (int j = 0; j < n_head; j++) {
            if (!to_test(i, j)) {
                continue;
            }
            std::cout << "Searching for layer " << i << " head " << j << std::endl;
            double total_time = 0;
            for (int k = sample_size; k < n_token; k++) {
                auto query = queries[i][j].data() + k * dim;
                auto s = std::chrono::high_resolution_clock::now();
                std::vector<uint32_t> res = graphs[i * n_head + j]->search(query, topK);
                auto e = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = e - s;
                total_time += diff.count();
            }
            std::cout << "Average search time: " << total_time / (n_token - sample_size) << "\n\n";
        }
    }

    std::cout << "END!!!" << std::endl;

    // delete
    for (int i = 0; i < graphs.size(); i++) {
        delete graphs[i];
    }

    return 0;
}
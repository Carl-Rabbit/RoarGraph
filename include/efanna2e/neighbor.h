//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_GRAPH_H
#define EFANNA2E_GRAPH_H

#include <cstddef>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <vector>
#define LIKELY(x) __builtin_expect(x, 1)
#define UNLIKELY(x) __builtin_expect(x, 0)
#include "util.h"

namespace efanna2e {

struct Neighbor {
    unsigned id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

    inline bool operator<(const Neighbor &other) const {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }
    
    inline bool operator==(const Neighbor &other) const { return (id == other.id); }
};

typedef std::lock_guard<std::mutex> LockGuard;
struct nhood {
    std::mutex lock;
    std::vector<Neighbor> pool;
    unsigned M;

    std::vector<unsigned> nn_old;
    std::vector<unsigned> nn_new;
    std::vector<unsigned> rnn_old;
    std::vector<unsigned> rnn_new;

    nhood() {}
    nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N) {
        M = s;
        nn_new.resize(s * 2);
        GenRandom(rng, &nn_new[0], (unsigned)nn_new.size(), N);
        nn_new.reserve(s * 2);
        pool.reserve(l);
    }

    nhood(const nhood &other) {
        M = other.M;
        std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
        nn_new.reserve(other.nn_new.capacity());
        pool.reserve(other.pool.capacity());
    }
    void insert(unsigned id, float dist) {
        LockGuard guard(lock);
        if (dist > pool.front().distance) return;
        for (unsigned i = 0; i < pool.size(); i++) {
            if (id == pool[i].id) return;
        }
        if (pool.size() < pool.capacity()) {
            pool.push_back(Neighbor(id, dist, true));
            std::push_heap(pool.begin(), pool.end());
        } else {
            std::pop_heap(pool.begin(), pool.end());
            pool[pool.size() - 1] = Neighbor(id, dist, true);
            std::push_heap(pool.begin(), pool.end());
        }
    }

    template <typename C>
    void join(C callback) const {
        for (unsigned const i : nn_new) {
            for (unsigned const j : nn_new) {
                if (i < j) {
                    callback(i, j);
                }
            }
            for (unsigned j : nn_old) {
                callback(i, j);
            }
        }
    }
};

struct SimpleNeighbor {
    unsigned id;
    float distance;

    SimpleNeighbor() = default;
    SimpleNeighbor(unsigned id, float distance) : id{id}, distance{distance} {}

    inline bool operator<(const SimpleNeighbor &other) const { return distance < other.distance; }
};
struct SimpleNeighbors {
    std::vector<SimpleNeighbor> pool;
};

static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
    // find the location to insert
    int left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
        memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
    if (addr[right].distance < nn.distance) {
        addr[K] = nn;
        return K;
    }
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance)
            right = mid;
        else
            left = mid;
    }
    // check equal ID

    while (left > 0) {
        if (addr[left].distance < nn.distance) break;
        if (addr[left].id == nn.id) return K + 1;
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id) return K + 1;
    memmove((char *)&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
}

class NeighborPriorityQueue {
   public:
    NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0) {}

    explicit NeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1) {}
    // : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1), candidates_id(capacity + 1) {}

    // Inserts the item ordered into the set up to the sets capacity.
    // The item will be dropped if it is the same id as an exiting
    // set item or it has a greated distance than the final
    // item in the set. The set cursor that is used to pop() the
    // next item will be set to the lowest index of an uncheck item
    void insert(const Neighbor &nbr) {
        if (_size == _capacity && _data[_size - 1] < nbr) {
            return;
        }

        size_t lo = 0, hi = _size;
        while (lo < hi) {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid]) {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            } else if (UNLIKELY(_data[mid].id == nbr.id)) {
                return;
            } else {
                lo = mid + 1;
            }
        }

        if (lo < _capacity) {
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
            // std::memmove(&candidates_id[lo + 1], &candidates_id[lo],
            //              (_size - lo) * sizeof(uint32_t));
        }

        _data[lo] = {nbr.id, nbr.distance, false};
        // candidates_id[lo] = nbr.id;

        if (_size < _capacity) {
            _size++;
        }
        if (lo < _cur) {
            _cur = lo;
        }
    }

    Neighbor closest_unexpanded() {
        _data[_cur].flag = true;
        size_t pre = _cur;
        while (_cur < _size && _data[_cur].flag) {
            _cur++;
        }
        return _data[pre];
    }

    bool has_unexpanded_node() const { return _cur < _size; }

    size_t size() const { return _size; }

    size_t capacity() const { return _capacity; }

    void reserve(size_t capacity) {
        if (capacity + 1 > _data.size()) {
            _data.resize(capacity + 1);
            // candidates_id.resize(capacity + 1);
        }
        _capacity = capacity;
    }

    Neighbor &operator[](size_t i) { return _data[i]; }

    Neighbor operator[](size_t i) const { return _data[i]; }

    void clear() {
        _size = 0;
        _cur = 0;
    }
    std::vector<Neighbor> &get_data() { return _data; }

    // std::vector<uint32_t> candidates_id;

   private:
    size_t _size, _capacity, _cur;
    std::vector<Neighbor> _data;
};

class NeighborDiffPriorityQueue {
   public:
    NeighborDiffPriorityQueue() : _size(0), _cur(0), _threshold(0) {}

    NeighborDiffPriorityQueue(float threshold, int capacity_factor) : _size(0), _cur(0), _threshold(threshold), _capacity_factor(capacity_factor), _data(0) {}

    void prepare_search() {
        if (_size == 0) {
            throw std::runtime_error("make sure pool has some points before calling prepare_search() (using insert()), then to start searching");
        }
        _drop_start = find_bsearch_dist(head_dist() + _threshold);
        _capacity = _drop_start * _capacity_factor;
    }

    inline size_t find_bsearch(const Neighbor &nbr) {
        size_t lo = 0, hi = _size;
        while (lo < hi) {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid]) {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            } else if (UNLIKELY(_data[mid].id == nbr.id)) {
                return lo;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    inline size_t find_bsearch_dist(const float dist) {
        return find_bsearch_dist(dist, _size);
    }

    inline size_t find_bsearch_dist(const float dist, size_t hi) {
        size_t lo = 0;
        while (lo < hi) {
            size_t mid = (lo + hi) >> 1;
            if (dist < _data[mid].distance) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    // Inserts the item ordered into the set up to the sets capacity.
    // The item will be dropped if it is the same id as an exiting
    // set item or it has a greated distance than the final
    // item in the set. The set cursor that is used to pop() the
    // next item will be set to the lowest index of an uncheck item
    int insert(const Neighbor &nbr) {
        size_t lo = find_bsearch(nbr);

        // insert at lo
        _data.insert(_data.begin() + lo, {nbr.id, nbr.distance, false});
        _size++;
        _cur = std::min(_cur, lo);

        return lo;
    }

    bool insert_and_mantain(const Neighbor &nbr) {
        size_t lo = find_bsearch(nbr);

        if (lo == 0) {
            // update head, some points we keeped are useless
            _data.insert(_data.begin(), {nbr.id, nbr.distance, false});
            // so push drop_start_ to the head
            // +1 as we insert a new point, +1 as hi in find_bsearch is exclusive
            _drop_start = find_bsearch_dist(head_dist() + _threshold, _drop_start + 2);
            // although drop_start_ updated, dynamic_capacity_ will not decrease
            _size = std::min(_capacity, _size + 1);
            _cur = 0;   // to explore it next
            return true;    // is important for sure
        }

        if (lo <= _drop_start) {
            if (nbr.distance <= head_dist() + _threshold) {
                // the point is imporant, e.g.
                // pool_dist=[1, 2, 3, 4], dist=2.4, threshold_=1.5, lo=2, drop_start_=2
                // it should at the left of drop_start
                _data.insert(_data.begin() + lo, {nbr.id, nbr.distance, false});
                _drop_start++;
                _capacity = _drop_start * _capacity_factor;
                _size = std::min(_capacity, _size + 1);
                _cur = std::min(lo, _cur);
                // now: pool_dist=[1, 2, 2.4, 3, 4, ?], drop_start_=3 (if factor=2)
                return true;
            } else {
                // the point is not imporant, but just on the border, e.g.,
                // pool_dist=[1,2,3,4], dist=2.6, threshold_=1.5, lo=2, drop_start_=2
                // it is located at drop_start_, so drop_start_ not changes
                _data.insert(_data.begin() + lo, {nbr.id, nbr.distance, false});
                _size = std::min(_capacity, _size + 1);
                _cur = std::min(lo, _cur);
                // now: pool_dist=[1, 2, 2.6, 3], drop_start_=2
                return false;
            }
        }

        if (lo < _capacity) {
            // this point is not important but will be in the list to explore
            _data.insert(_data.begin() + lo, {nbr.id, nbr.distance, false});
            _size = std::min(_capacity, _size + 1);
            _cur = std::min(lo, _cur);
            return false;
        }

        // this point will not be considered
        return false;
    }

    Neighbor closest_unexpanded() {
        _data[_cur].flag = true;
        size_t pre = _cur;
        while (_cur < _size && _data[_cur].flag) {
            _cur++;
        }
        return _data[pre];
    }

    bool has_unexpanded_node() const { return _cur < _size; }

    size_t size() const { return _size; }

    Neighbor &operator[](size_t i) { return _data[i]; }

    Neighbor operator[](size_t i) const { return _data[i]; }

    void clear() {
        _size = 0;
        _cur = 0;
    }
    std::vector<Neighbor> &get_data() { return _data; }

    float head_dist() { return _data[0].distance; }
    float tail_dist() { return _data[_size - 1].distance; }

    // std::vector<uint32_t> candidates_id;

   private:
    size_t _size, _cur;
    float _threshold;
    int _capacity_factor;
    std::vector<Neighbor> _data;
    size_t _drop_start;
    size_t _capacity;
};

class NeighborMaxsumPriorityQueue {
   public:
    NeighborMaxsumPriorityQueue() : _size(0), _cur(0), _capacity(0), _data(0) {}

    NeighborMaxsumPriorityQueue(int capacity) : _size(0), _cur(0), _capacity(capacity), _data(0) {
        _data.reserve(capacity + 1);
    }

    inline size_t find_bsearch(const Neighbor &nbr) {
        size_t lo = 0, hi = _size;
        while (lo < hi) {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid]) {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            } else if (UNLIKELY(_data[mid].id == nbr.id)) {
                return lo;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    inline size_t find_bsearch_dist(const float dist) {
        return find_bsearch_dist(dist, _size);
    }

    inline size_t find_bsearch_dist(const float dist, size_t hi) {
        size_t lo = 0;
        while (lo < hi) {
            size_t mid = (lo + hi) >> 1;
            // std::cout << "mid: " << mid << ", size: " << _size << ", capacity: " << _capacity << std::endl;
            if (dist < _data[mid].distance) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    // Notice that this insert will enlarge the size if need
    int insert(const Neighbor &nbr) {
        size_t lo = find_bsearch(nbr);

        // insert at lo
        _data.insert(_data.begin() + lo, {nbr.id, nbr.distance, false});
        _size++;
        _capacity = std::max(_size + 1, _capacity);     // +1 as use memmove
        _cur = std::min(_cur, lo);

        return lo;
    }

    bool insert_at(const Neighbor &nbr, size_t index) {
        if (_capacity <= index + 1) {
            return false;
        }
        // std::cout << "insert_at: index=" << index << ", capacity=" << _capacity << ", size=" << _size << std::endl;
        _data.insert(_data.begin() + index, {nbr.id, nbr.distance, false});
        // std::memmove(&_data[index + 1], &_data[index], (_size - index) * sizeof(Neighbor));
        // _data[index] = nbr;
        if (_size + 1 < _capacity) {        // +1 as use memmove
            _size++;
        }
        _cur = std::min(_cur, index);
        return true;
    }

    Neighbor closest_unexpanded() {
        _data[_cur].flag = true;
        size_t pre = _cur;
        while (_cur < _size && _data[_cur].flag) {
            _cur++;
        }
        return _data[pre];
    }

    bool has_unexpanded_node() const { return _cur < _size; }

    size_t size() const { return _size; }

    Neighbor &operator[](size_t i) { return _data[i]; }

    Neighbor operator[](size_t i) const { return _data[i]; }

    void reserve_if_not_enough(size_t capacity) {
        capacity += 1;      // +1 as use memmove
        if (capacity <= _capacity) {
            return;
        }
        // std::cout << "reserve_if_not_enough: capacity=" << capacity << ", size=" << _size << std::endl;
        _capacity = capacity;
    }

    std::vector<Neighbor> &get_data() { return _data; }

    float head_dist() { return _data[0].distance; }
    float tail_dist() { return _data[_size - 1].distance; }
    float dist(size_t idx) { return _data[idx].distance; }

    // std::vector<uint32_t> candidates_id;

   private:
    size_t _size, _cur;
    size_t _capacity;
    std::vector<Neighbor> _data;
};

}  // namespace efanna2e

#endif  // EFANNA2E_GRAPH_H
